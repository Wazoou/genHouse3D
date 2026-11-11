"""Geometric clean-up of generated plans.

Only the last steps exist so far (milestone 3): attaching doors to the walls shared by the
rooms that the requested graph connects, and the validity verdict. Snapping, overlap removal
and gap closing come with the trained model (milestone 5).
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from housegen.data.geometry import straight_pieces
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
    shapes = [Room(r.type, r.polygon).shape() for r in rooms]
    doors: list[Door] = []
    open_edges: list[tuple[int, int]] = []
    for i, j, kind in graph.edges:
        if not (shapes[i].is_valid and shapes[j].is_valid):
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
        if not shapes[i].is_valid:
            continue
        others = [s.buffer(tol) if tol > 0 else s.exterior for k, s in enumerate(shapes) if k != i]
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
