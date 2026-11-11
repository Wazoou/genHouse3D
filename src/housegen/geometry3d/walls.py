"""2D wall layout of a plan: footprint, outer walls and inner partitions.

All 2D work is done in grid units (integers for the plan itself), which keeps the topology
tests exact; coordinates are scaled to metres when the meshes are built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from housegen.data.geometry import straight_pieces
from housegen.data.schema import Plan
from housegen.geometry3d.mesh import polygons

OUTDOOR_TYPES = frozenset({"balcony"})


def fill_holes(geom) -> Polygon | MultiPolygon:
    return unary_union([Polygon(p.exterior) for p in polygons(geom)])


@dataclass
class Layout:
    """Plan geometry in grid units, centred on the origin."""

    plan: Plan
    unit: float  # metres per grid unit
    origin: np.ndarray  # grid coordinates of the layout origin
    rooms: list[Polygon]
    interior: Polygon | MultiPolygon  # union of the indoor rooms
    outer: Polygon | MultiPolygon  # outer face of the outer walls, holes filled
    walls: Polygon | MultiPolygon  # footprint of all walls
    t_ext: float  # wall thicknesses in grid units
    t_int: float

    @property
    def types(self) -> list[str]:
        return [r.type for r in self.plan.rooms]

    def is_outdoor(self, room: int) -> bool:
        return self.plan.rooms[room].type in OUTDOOR_TYPES

    def to_local(self, point) -> np.ndarray:
        return np.asarray(point, dtype=float) - self.origin


def build_layout(plan: Plan, cfg: dict) -> Layout:
    """Wall footprint of ``plan``.

    Outer walls are built outwards from the edges of the indoor footprint; partitions are
    centred on the edges shared by two indoor rooms, except between rooms that are open onto
    each other. Voids enclosed by the rooms (shafts) become solid wall.
    """
    unit = plan.meters_per_unit
    t_ext = cfg["walls"]["exterior_thickness"] / unit
    t_int = cfg["walls"]["interior_thickness"] / unit
    pts = np.concatenate([np.asarray(r.polygon, dtype=float) for r in plan.rooms])
    origin = np.floor((pts.min(axis=0) + pts.max(axis=0)) / 2.0)
    rooms = [affinity.translate(r.shape(), -origin[0], -origin[1]) for r in plan.rooms]
    indoor = [k for k, r in enumerate(plan.rooms) if r.type not in OUTDOOR_TYPES]
    if not indoor:
        raise ValueError("plan has no indoor room")
    interior = unary_union([rooms[k] for k in indoor])
    outer = fill_holes(interior).buffer(t_ext, join_style="mitre")
    open_pairs = {(i, j) for i, j, kind in plan.edges() if kind == "open"}
    partitions = []
    for a, i in enumerate(indoor):
        for j in indoor[a + 1 :]:
            if (i, j) in open_pairs:
                continue
            shared = rooms[i].exterior.intersection(rooms[j].exterior)
            partitions += [
                piece.buffer(t_int / 2.0, cap_style="square") for piece in straight_pieces(shared)
            ]
    walls = unary_union([outer.difference(interior), *partitions]).intersection(outer)
    return Layout(plan, unit, origin, rooms, interior, outer, walls, t_ext, t_int)
