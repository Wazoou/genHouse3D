"""Adapter for the Swiss Dwellings dataset (Archilyse AG, v3.0.0, CC BY 4.0).

Source: https://doi.org/10.5281/zenodo.7788422 (``geometries.csv``). It is the parent dataset
of Modified Swiss Dwellings (MSD). MSD works at the scale of whole building floors (20 to 70
rooms); here every *apartment* becomes one plan, which matches the 4 to 10 room dwellings this
project generates.

Each CSV row is one entity (area, wall, opening, feature) with a WKT polygon in metres, in a
site-local frame. Areas are net room outlines, separated by the wall thickness. The adapter
rotates each apartment onto the axes, rectifies its rooms, closes the wall gaps so that rooms
share edges, snaps everything to the metric grid and re-attaches doors to the shared walls.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from housegen.data.geometry import (
    clean_ring,
    dominant_angle,
    place_on_wall,
    rectify_ring,
    snap_walls,
    straight_pieces,
)
from housegen.data.schema import GRID_SIZE, METERS_PER_UNIT, OUTSIDE, Door, Plan, Room

SOURCE = "swiss_dwellings"

# Raw ``entity_subtype`` of areas -> unified room type. ``ROOM`` is the generic habitable room
# of the dataset (bedrooms, studies); MSD maps it to "Bedroom" too.
DEFAULT_TYPE_MAP: dict[str, str] = {
    "LIVING_ROOM": "living",
    "LIVING_DINING": "living",
    "KITCHEN": "kitchen",
    "KITCHEN_DINING": "kitchen",
    "DINING": "dining",
    "BEDROOM": "bedroom",
    "ROOM": "bedroom",
    "BATHROOM": "bathroom",
    "CORRIDOR": "corridor",
    "CORRIDORS_AND_HALLS": "corridor",
    "STOREROOM": "storage",
    "BALCONY": "balcony",
    "LOGGIA": "balcony",
    "TERRACE": "balcony",
}
# Areas that are not rooms: dropped without rejecting the apartment.
DEFAULT_IGNORED: frozenset[str] = frozenset(
    {"SHAFT", "VOID", "LIGHTWELL", "OUTDOOR_VOID", "AIR", "ELEVATOR"}
)


@dataclass
class SwissDwellingsConfig:
    type_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPE_MAP))
    ignored_subtypes: frozenset[str] = DEFAULT_IGNORED
    max_wall_m: float = 0.7  # widest wall closed between two facing room faces
    align_tol_m: float = 0.1  # wall faces closer than this are put on one line
    max_trim_m2: float = 0.25  # largest residual overlap trimmed off the bigger room
    angle_tol_deg: float = 12.0  # edges further from both axes make a room non rectilinear
    max_diagonal_m: float = 0.15  # ... unless they are shorter than this
    door_touch_m: float = 0.05  # a door belongs to the rooms closer than this
    door_max_shift_m: float = 0.6  # max distance between a raw door and its snapped wall
    door_width_m: tuple[float, float] = (0.6, 1.6)  # clamp of door widths
    min_door_wall_m: float = 0.5  # shortest wall piece that can carry a door
    open_touch_m: float = 0.03  # rooms closer than this are open onto each other...
    open_min_length_m: float = 0.5  # ... when they touch along at least this length


@dataclass
class RawApartment:
    site_id: str
    building_id: str
    floor_id: str
    apartment_id: str
    rooms: list[tuple[str, Polygon]]  # (raw subtype, outline in metres)
    doors: list[tuple[bool, Polygon]]  # (is entrance door, footprint) candidates on the floor


def iter_raw_apartments(csv_path: str | Path) -> Iterator[RawApartment | str]:
    """Yield the residential apartments of ``geometries.csv``.

    Apartments spread over several floors are reported as the string ``"multi_floor"``
    instead (one per apartment), since the project only handles single-storey dwellings.
    """
    csv.field_size_limit(2**31 - 1)
    areas: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    doors: dict[str, dict[str, bool]] = defaultdict(dict)
    floors_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kind, subtype, floor = row["entity_type"], row["entity_subtype"], row["floor_id"]
            if kind == "area":
                if row["unit_usage"] == "RESIDENTIAL" and row["apartment_id"]:
                    key = (row["site_id"], row["apartment_id"])
                    floors_of[key].add(floor)
                    areas[floor].append(
                        (
                            row["site_id"],
                            row["building_id"],
                            row["apartment_id"],
                            subtype,
                            row["geometry"],
                        )
                    )
            elif kind == "opening" and subtype in ("DOOR", "ENTRANCE_DOOR"):
                # The same door is listed once per unit it touches: keep one copy.
                doors[floor][row["geometry"]] = subtype == "ENTRANCE_DOOR"
    multi = {key for key, floors in floors_of.items() if len(floors) > 1}
    for _ in multi:
        yield "multi_floor"
    for floor, rows in areas.items():
        door_flags = list(doors.get(floor, {}).values())
        door_geoms = shapely.from_wkt(list(doors.get(floor, {})))
        tree = shapely.STRtree(door_geoms) if len(door_geoms) else None
        geoms = shapely.from_wkt([r[4] for r in rows])
        by_apartment: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            if (r[0], r[2]) not in multi:
                by_apartment[(r[0], r[2])].append(i)
        for (site, apartment), idx in by_apartment.items():
            rooms = [(rows[i][3], geoms[i]) for i in idx]
            near: list[tuple[bool, Polygon]] = []
            if tree is not None:
                hull = unary_union([g for _, g in rooms]).buffer(0.2)
                near = [(door_flags[k], door_geoms[k]) for k in sorted(tree.query(hull))]
            yield RawApartment(site, rows[idx[0]][1], floor, apartment, rooms, near)


def _add_door(doors: list[Door], door: Door) -> None:
    """Append ``door`` unless the same rooms already have a door at about the same place."""
    for other in doors:
        if other.rooms == door.rooms and other.line().distance(door.line()) < 1.0:
            return
    doors.append(door)


def convert_apartment(raw: RawApartment, cfg: SwissDwellingsConfig) -> Plan | str:
    """Convert one raw apartment into a :class:`Plan`, or return the rejection reason."""
    kept: list[tuple[str, str, Polygon]] = []
    for subtype, geom in raw.rooms:
        if subtype in cfg.ignored_subtypes:
            continue
        if subtype not in cfg.type_map:
            return "unsupported_room_type"
        if geom.geom_type != "Polygon" or geom.is_empty:
            return "invalid_raw_geometry"
        kept.append((cfg.type_map[subtype], subtype, geom))
    if not kept:
        return "no_room"
    # Many apartments only use the generic ROOM label: without an annotated living room, the
    # largest generic room is taken as the living room.
    living_inferred = False
    generic = [k for k, (_, subtype, _) in enumerate(kept) if subtype == "ROOM"]
    if generic and all(room_type != "living" for room_type, _, _ in kept):
        k = max(generic, key=lambda k: kept[k][2].area)
        kept[k] = ("living", "ROOM", kept[k][2])
        living_inferred = True

    # 1. Rotate the apartment so that its walls follow the axes.
    angle = dominant_angle([g for _, _, g in kept])
    origin = unary_union([g for _, _, g in kept]).envelope.centroid

    def rotate(geom):
        return affinity.rotate(geom, -angle, origin=origin, use_radians=True)

    raw_polys = [rotate(g) for _, _, g in kept]

    # 2. Rectify each room, 3. close the wall gaps.
    rings = []
    for poly in raw_polys:
        ring = rectify_ring(poly, cfg.angle_tol_deg, cfg.max_diagonal_m)
        if ring is None:
            return "non_rectilinear"
        rings.append(ring)
    rings = snap_walls(rings, cfg.max_wall_m, align_tol=cfg.align_tol_m)

    # 4. Snap to the grid, centred in the GRID_SIZE x GRID_SIZE canvas.
    mpu = METERS_PER_UNIT
    lo = np.min([r.min(axis=0) for r in rings], axis=0)
    units = [np.rint((r - lo) / mpu).astype(int) for r in rings]
    size = np.max([u.max(axis=0) for u in units], axis=0)
    if size.max() > GRID_SIZE - 1:
        return "too_large"
    offset = (GRID_SIZE - 1 - size) // 2
    grid_rings: list[np.ndarray] = []
    for u in units:
        ring = clean_ring(u + offset)
        if ring is None or not Polygon(ring).is_valid:
            return "degenerate_room"
        grid_rings.append(ring)
    # Residual overlaps: a small one is trimmed off the bigger room, a large one is an error.
    shapes = [Polygon(r) for r in grid_rings]
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            overlap = shapes[i].intersection(shapes[j]).area
            if overlap <= 1e-3:
                continue
            big, small = (i, j) if shapes[i].area >= shapes[j].area else (j, i)
            if overlap > cfg.max_trim_m2 / mpu**2 or overlap > 0.05 * shapes[small].area:
                return "overlapping_rooms"
            trimmed = shapes[big].difference(shapes[small])
            if trimmed.geom_type != "Polygon" or trimmed.interiors:
                return "overlapping_rooms"
            ring = clean_ring(np.rint(np.asarray(trimmed.exterior.coords)[:-1]).astype(int))
            if ring is None or not Polygon(ring).is_valid:
                return "overlapping_rooms"
            grid_rings[big], shapes[big] = ring, Polygon(ring)
    if any(len(ring) % 2 for ring in grid_rings):  # a rectilinear ring has an even corner count
        return "degenerate_room"
    rooms = [
        Room(room_type, tuple((int(x), int(y)) for x, y in ring))
        for (room_type, _, _), ring in zip(kept, grid_rings, strict=True)
    ]

    def to_grid(geom):
        return affinity.affine_transform(
            geom, [1 / mpu, 0, 0, 1 / mpu, offset[0] - lo[0] / mpu, offset[1] - lo[1] / mpu]
        )

    # 5. Doors: attach each raw door to the rooms it touches, then lay it on their wall.
    exteriors = [s.exterior for s in shapes]
    doors: list[Door] = []
    max_shift, min_wall = cfg.door_max_shift_m / mpu, cfg.min_door_wall_m / mpu
    for is_entrance, geom in raw.doors:
        geom = rotate(geom)
        if min(p.distance(geom) for p in raw_polys) >= cfg.door_touch_m:
            continue  # door of another apartment on the same floor
        box_ = geom.minimum_rotated_rectangle
        if box_.geom_type != "Polygon":
            continue
        # A door crosses its wall: the rooms it connects lie just beyond its two long sides.
        rect = np.asarray(box_.exterior.coords)[:4]
        e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
        l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
        if min(l1, l2) <= 0:
            continue
        width, thickness, across = (l1, l2, e2 / l2) if l1 >= l2 else (l2, l1, e1 / l1)
        middle = rect.mean(axis=0)
        sides: list[int] = []
        for sign in (1.0, -1.0):
            probe = Point(middle + sign * (thickness / 2 + cfg.door_touch_m) * across)
            dist = [p.distance(probe) for p in raw_polys]
            k = int(np.argmin(dist))
            if dist[k] < 2 * cfg.door_touch_m:
                sides.append(k)
        width = min(max(width, cfg.door_width_m[0]), cfg.door_width_m[1]) / mpu
        center = to_grid(Point(middle))
        if len(sides) == 2 and sides[0] != sides[1]:
            i, j = sorted(sides)
            pieces = straight_pieces(exteriors[i].intersection(exteriors[j]))
            segment = place_on_wall(pieces, center, width, max_shift, min_wall)
            if segment is None:
                return "door_without_shared_wall"
            _add_door(doors, Door((i, j), segment))
        elif len(sides) == 1 and is_entrance:
            i = sides[0]
            others = unary_union([e for k, e in enumerate(exteriors) if k != i])
            pieces = straight_pieces(exteriors[i].difference(others))
            segment = place_on_wall(pieces, center, width, max_shift, min_wall)
            if segment is not None:
                _add_door(doors, Door((i, OUTSIDE), segment))
        # An inner door with a room on one side only leads out of the apartment: ignored.

    # 6. Open connections: rooms touching without a wall in between.
    linked = {tuple(d.rooms) for d in doors if not d.is_entrance}
    open_edges: list[tuple[int, int]] = []
    tol = cfg.open_touch_m
    for i in range(len(raw_polys)):
        for j in range(i + 1, len(raw_polys)):
            if (i, j) in linked or raw_polys[i].distance(raw_polys[j]) >= tol:
                continue
            band = raw_polys[i].buffer(tol).intersection(raw_polys[j].buffer(tol))
            shared = exteriors[i].intersection(exteriors[j])
            if band.area / (2 * tol) >= cfg.open_min_length_m and shared.length > 0:
                open_edges.append((i, j))

    return Plan(
        id=f"sd_{raw.site_id}_{raw.apartment_id[:12]}",
        source=SOURCE,
        group=f"sd_site_{raw.site_id}",
        rooms=tuple(rooms),
        doors=tuple(doors),
        open_edges=tuple(open_edges),
        meta={
            "site_id": raw.site_id,
            "building_id": raw.building_id,
            "floor_id": raw.floor_id,
            "apartment_id": raw.apartment_id,
            "rotation_deg": round(math.degrees(angle), 3),
            "raw_types": [subtype for _, subtype, _ in kept],
            "living_inferred": living_inferred,
        },
    )


def iter_plans(
    csv_path: str | Path, cfg: SwissDwellingsConfig | None = None, rejected: Counter | None = None
) -> Iterator[Plan]:
    """Yield converted plans; rejection reasons are tallied in ``rejected`` when given."""
    cfg = cfg or SwissDwellingsConfig()
    for raw in iter_raw_apartments(csv_path):
        result = raw if isinstance(raw, str) else convert_apartment(raw, cfg)
        if isinstance(result, str):
            if rejected is not None:
                rejected[result] += 1
        else:
            yield result
