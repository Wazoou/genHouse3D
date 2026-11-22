"""Geometric clean-up of generated plans (brief, section 5).

:func:`clean_plan` applies, in order:

1. repair: self-intersecting rooms are made valid (their largest part is kept);
2. Manhattan straightening: nearly horizontal or vertical walls become exactly so;
3. gap closing and alignment: facing walls of neighbouring rooms less than ``max_gap_m``
   apart (or overlapping by less than ``max_overlap_m``) move onto a common line, nearly
   collinear walls are aligned, then corners are rounded to the grid;
4. overlap removal: where two rooms still overlap, the smaller one keeps its shape and the
   larger one is cropped (unless that would leave a hole: the overlap then stays);
5. doors on the walls shared by the rooms the graph connects, the entrance door on an outer
   wall of the entrance room (or of the living room when it has none).

:func:`validity` gives the verdict. Metrics are always reported before and after clean-up.

Usage: ``python -m housegen.postprocess --input raw.jsonl.gz --output clean.jsonl.gz``
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from housegen.data.geometry import clean_ring, rectify_ring, snap_walls, straight_pieces
from housegen.data.schema import GRID_SIZE, OUTSIDE, ROOM_TYPES, Door, Plan, Room
from housegen.graphs import RoomGraph


def shared_wall(a: Polygon, b: Polygon, tol: float = 0.0) -> list[LineString]:
    """Straight pieces of the boundary of ``a`` lying on (or within ``tol`` of) ``b``."""
    if tol <= 0:
        return straight_pieces(a.exterior.intersection(b.exterior))
    return straight_pieces(a.exterior.intersection(b.buffer(tol)))


def _door_on(pieces: list[LineString], width: float, min_wall: float):
    """Door footprint centred on the longest wall piece, or ``None`` if none is long enough."""
    pieces = [p for p in pieces if p.length >= min_wall]
    if not pieces:
        return None
    piece = max(pieces, key=lambda p: p.length)
    half = min(width, piece.length) / 2.0
    a = piece.interpolate(piece.length / 2.0 - half)
    b = piece.interpolate(piece.length / 2.0 + half)
    return (round(a.x, 2), round(a.y, 2)), (round(b.x, 2), round(b.y, 2))


def _usable(room: Room) -> Polygon | None:
    if len(room.polygon) < 3:
        return None
    shape = room.shape()
    return shape if shape.is_valid and shape.area > 0 else None


def attach_doors(
    plan_id: str,
    rooms: Sequence[Room],
    graph: RoomGraph,
    door_width: float = 9.0,
    entrance_width: float = 10.0,
    min_wall: float = 5.0,
    tol: float = 0.0,
    source: str = "generated",
    group: str = "",
    meta: dict | None = None,
) -> Plan:
    """Plan made of ``rooms`` with the connections of ``graph`` that the geometry allows.

    A requested connection is kept when the two rooms share a wall piece of at least
    ``min_wall`` grid units: a door is centred on the longest such piece (or the rooms are
    recorded as open onto each other, for an ``open`` connection). The entrance door goes on
    the longest outer wall of an entrance room. Connections the geometry cannot carry are
    simply absent from the result: that is what the compatibility metric counts.
    """
    shapes = [_usable(r) for r in rooms]
    doors: list[Door] = []
    open_edges: list[tuple[int, int]] = []
    for i, j, kind in graph.edges:
        if shapes[i] is None or shapes[j] is None:
            continue
        pieces = shared_wall(shapes[i], shapes[j], tol)
        if kind == "open":
            if any(p.length >= min_wall for p in pieces):
                open_edges.append((i, j))
        else:
            segment = _door_on(pieces, door_width, min_wall)
            if segment is not None:
                doors.append(Door((i, j), segment))
    for i in graph.entrances:
        if shapes[i] is None:
            continue
        others = [
            s.buffer(tol) if tol > 0 else s.exterior
            for k, s in enumerate(shapes)
            if k != i and s is not None
        ]
        free = shapes[i].exterior.difference(unary_union(others)) if others else shapes[i].exterior
        segment = _door_on(straight_pieces(free), entrance_width, min_wall)
        if segment is not None:
            doors.append(Door((i, OUTSIDE), segment))
            break
    return Plan(
        plan_id, source, group, tuple(rooms), tuple(doors), tuple(open_edges), meta=meta or {}
    )


# Smallest plausible room areas, in square metres, by type (see configs/eval.yaml).
DEFAULT_MIN_AREA: dict[str, float] = dict.fromkeys(ROOM_TYPES, 0.0)


def validity(plan: Plan, min_area: dict[str, float] | None = None) -> str:
    """``"valid"`` or the first reason why ``plan`` is not usable as a dwelling.

    Checks, in order: simple polygons inside the grid, no overlap between rooms, an
    entrance door, every room reachable from the entrance through doors and openings,
    plausible room areas.
    """
    min_area = min_area or DEFAULT_MIN_AREA
    shapes = []
    for room in plan.rooms:
        shape = room.shape() if len(room.polygon) >= 3 else Polygon()
        if shape.is_empty or not shape.is_valid or shape.area <= 0:
            return "invalid_polygon"
        x0, y0, x1, y1 = shape.bounds
        if x0 < 0 or y0 < 0 or x1 > GRID_SIZE - 1 or y1 > GRID_SIZE - 1:
            return "outside_grid"
        shapes.append(shape)
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i].intersection(shapes[j]).area > 1e-3:
                return "overlap"
    entrances = plan.entrance_rooms()
    if not entrances:
        return "no_entrance"
    graph = plan.graph()
    reachable = set().union(*(nx.node_connected_component(graph, e) for e in entrances))
    if len(reachable) < len(plan.rooms):
        return "unreachable_room"
    for i, room in enumerate(plan.rooms):
        if plan.room_area_m2(i) < min_area.get(room.type, 0.0):
            return "implausible_area"
    return "valid"


# ---------------------------------------------------------------- clean-up


@dataclass
class CleanupConfig:
    """Thresholds of :func:`clean_plan`, in metres unless stated otherwise."""

    angle_tol_deg: float = 20.0  # walls closer than this to an axis are straightened...
    max_diagonal_m: float = 0.3  # ...and shorter slanted pieces are absorbed
    max_area_change: float = 0.2  # straightening may change a room's area by this share
    max_gap_m: float = 0.8  # facing walls closer than this are joined... (tuned on val)
    max_overlap_m: float = 0.4  # ...as are facing walls overlapping by less than this...
    min_contact_m: float = 0.3  # ...if they face each other over at least this length
    align_m: float = 0.05  # nearly collinear walls closer than this are aligned (below one
    # grid unit: real plans have legitimate 10 cm offsets between walls)
    block_share: float = 0.5  # walls are not joined across another room covering this share
    door_width_m: float = 0.9
    entrance_width_m: float = 1.0
    min_door_wall_m: float = 0.5


def requested_graph(plan: Plan) -> RoomGraph:
    """The graph a generated plan was asked for, as stored in its meta by ``housegen.sample``."""
    meta = plan.meta
    return RoomGraph(
        id=meta.get("condition", plan.id),
        types=tuple(r.type for r in plan.rooms),
        corners=tuple(len(r.polygon) for r in plan.rooms),
        edges=tuple((int(i), int(j), str(kind)) for i, j, kind in meta["edges"]),
        entrances=tuple(int(i) for i in meta["entrances"]),
    )


def _largest_polygon(geom) -> Polygon | None:
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon) and g.area > 0]
    if not parts:
        return None
    return Polygon(max(parts, key=lambda g: g.area).exterior)


def _on_grid(points: np.ndarray) -> Polygon | None:
    ring = clean_ring(np.rint(points).astype(int))
    if ring is None:
        return None
    shape = Polygon(ring)
    return shape if shape.is_valid and shape.area > 0 else None


def clean_plan(plan: Plan, graph: RoomGraph, cfg: CleanupConfig | None = None) -> tuple[Plan, dict]:
    """Cleaned copy of a generated ``plan`` and counters describing what was changed.

    Room order and types are kept, so the result still matches ``graph``; a room that
    collapses gets an empty polygon and the plan is then judged invalid.
    """
    cfg = cfg or CleanupConfig()
    unit = plan.meters_per_unit
    stats = Counter(repaired=0, not_rectilinear=0, cropped=0, enclosed=0, collapsed=0)

    # 1. repair
    outlines: list[Polygon | None] = []
    for room in plan.rooms:
        shape = Polygon(room.polygon) if len(room.polygon) >= 3 else Polygon()
        if not shape.is_valid:
            shape = _largest_polygon(make_valid(shape))
            stats["repaired"] += 1
        outlines.append(shape if shape is not None and shape.area > 0 else None)

    # 2. straighten, 3. close gaps and align (rectilinear rooms only)
    rings: list[np.ndarray | None] = []
    for shape in outlines:
        ring = None
        if shape is not None:
            ring = rectify_ring(
                shape, cfg.angle_tol_deg, cfg.max_diagonal_m / unit, simplify_tol=0.3,
                max_area_change=cfg.max_area_change,
            )  # fmt: skip
            if ring is None:
                stats["not_rectilinear"] += 1
        rings.append(ring)
    straight = [k for k, r in enumerate(rings) if r is not None]
    if straight:
        snapped = snap_walls(
            [rings[k] for k in straight], max_wall=cfg.max_gap_m / unit,
            min_overlap=cfg.min_contact_m / unit, align_tol=cfg.align_m / unit,
            max_overlap=cfg.max_overlap_m / unit, block_share=cfg.block_share,
        )  # fmt: skip
        for k, ring in zip(straight, snapped, strict=True):
            rings[k] = ring
    shapes: list[Polygon | None] = []
    for k, ring in enumerate(rings):
        if ring is None and outlines[k] is not None:  # not rectilinear: keep it, on the grid
            ring = np.asarray(outlines[k].exterior.coords)[:-1]
        shapes.append(None if ring is None else _on_grid(ring))

    # 4. overlaps: the smaller room wins
    order = sorted((k for k, s in enumerate(shapes) if s is not None), key=lambda k: shapes[k].area)
    for a, small in enumerate(order):
        for big in order[a + 1 :]:
            if shapes[small] is None or shapes[big] is None:
                continue
            if shapes[big].intersection(shapes[small]).area <= 1e-6:
                continue
            remainder = shapes[big].difference(shapes[small])
            if any(
                getattr(part, "interiors", ()) for part in getattr(remainder, "geoms", [remainder])
            ):
                stats["enclosed"] += 1  # cropping would leave a hole: keep the overlap
                continue
            cropped = _largest_polygon(remainder)
            shapes[big] = None if cropped is None else _on_grid(
                np.asarray(cropped.exterior.coords)[:-1]
            )  # fmt: skip
            stats["cropped"] += 1

    rooms = []
    for room, shape in zip(plan.rooms, shapes, strict=True):
        if shape is None:
            stats["collapsed"] += 1
            rooms.append(Room(room.type, ()))
        else:
            ring = np.asarray(shape.exterior.coords)[:-1]
            rooms.append(Room(room.type, tuple((int(x), int(y)) for x, y in ring)))

    # 5. doors
    meta = plan.meta | {"cleanup": dict(stats)}

    def with_entrances(entrances: tuple[int, ...]) -> Plan:
        return attach_doors(
            plan.id, rooms, dataclasses.replace(graph, entrances=entrances),
            cfg.door_width_m / unit, cfg.entrance_width_m / unit, cfg.min_door_wall_m / unit,
            source=plan.source, group=plan.group, meta=meta,
        )  # fmt: skip

    living = tuple(k for k, r in enumerate(rooms) if r.type == "living")
    cleaned = with_entrances(graph.entrances or living)
    if not cleaned.entrance_rooms() and graph.entrances and living:
        fallback = with_entrances(living)
        if fallback.entrance_rooms():
            cleaned = fallback
    return cleaned, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from housegen.data.schema import iter_plans, write_plans

    totals: Counter = Counter()

    def cleaned():
        for plan in iter_plans(args.input):
            result, stats = clean_plan(plan, requested_graph(plan))
            totals.update(stats)
            yield result

    count = write_plans(args.output, cleaned())
    print(f"cleaned {count} plans: {dict(totals)}")


if __name__ == "__main__":
    main()
