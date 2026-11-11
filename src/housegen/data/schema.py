"""Unified floorplan format shared by every dataset adapter, the model and the 3D pipeline.

A plan is a planar partition: rooms are polygons that share edges (walls have no thickness
here, they are re-created with real thicknesses by ``housegen.geometry3d``). Coordinates live
on a fixed metric grid of ``GRID_SIZE`` x ``GRID_SIZE`` units, ``METERS_PER_UNIT`` metres per
unit, so that the real-world scale is never lost.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.geometry import Point as ShapelyPoint

ROOM_TYPES: tuple[str, ...] = (
    "living",
    "kitchen",
    "dining",
    "bedroom",
    "bathroom",
    "corridor",
    "storage",
    "balcony",
)
EDGE_KINDS: tuple[str, ...] = ("door", "open")
GRID_SIZE = 256
METERS_PER_UNIT = 0.1
OUTSIDE = -1  # pseudo room index used by the entrance door

Point = tuple[float, float]


@dataclass(frozen=True)
class Room:
    """A typed room: counter-clockwise ring of corners in grid units, not closed."""

    type: str
    polygon: tuple[Point, ...]

    def shape(self) -> Polygon:
        return Polygon(self.polygon)


@dataclass(frozen=True)
class Door:
    """A door lying on a wall: ``segment`` is its footprint along the wall, in grid units.

    ``rooms`` holds the two room indices it connects; the entrance door uses ``OUTSIDE`` as
    its second index.
    """

    rooms: tuple[int, int]
    segment: tuple[Point, Point]

    @property
    def is_entrance(self) -> bool:
        return self.rooms[1] == OUTSIDE

    def line(self) -> LineString:
        return LineString(self.segment)


@dataclass(frozen=True)
class Plan:
    """One single-storey dwelling in the unified format."""

    id: str
    source: str
    group: str  # leakage group: plans sharing a group always land in the same split
    rooms: tuple[Room, ...]
    doors: tuple[Door, ...] = ()
    open_edges: tuple[tuple[int, int], ...] = ()  # rooms connected without a door
    meters_per_unit: float = METERS_PER_UNIT
    meta: dict = field(default_factory=dict)

    def edges(self) -> list[tuple[int, int, str]]:
        """Room connections ``(i, j, kind)`` with ``i < j``; a door wins over an opening."""
        kinds: dict[tuple[int, int], str] = {}
        for i, j in self.open_edges:
            kinds[(min(i, j), max(i, j))] = "open"
        for door in self.doors:
            if not door.is_entrance:
                i, j = door.rooms
                kinds[(min(i, j), max(i, j))] = "door"
        return [(i, j, kind) for (i, j), kind in sorted(kinds.items())]

    def graph(self) -> nx.Graph:
        """Room graph (bubble diagram): typed nodes, edges for door / open connections."""
        g = nx.Graph()
        for i, room in enumerate(self.rooms):
            g.add_node(i, type=room.type, corners=len(room.polygon))
        for i, j, kind in self.edges():
            g.add_edge(i, j, kind=kind)
        return g

    def entrance_rooms(self) -> list[int]:
        return sorted({d.rooms[0] for d in self.doors if d.is_entrance})

    def room_area_m2(self, index: int) -> float:
        return self.rooms[index].shape().area * self.meters_per_unit**2

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "group": self.group,
            "meters_per_unit": self.meters_per_unit,
            "rooms": [
                {"type": r.type, "polygon": [list(p) for p in r.polygon]} for r in self.rooms
            ],
            "doors": [
                {"rooms": list(d.rooms), "segment": [list(p) for p in d.segment]}
                for d in self.doors
            ],
            "open_edges": [list(e) for e in self.open_edges],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(
            id=d["id"],
            source=d["source"],
            group=d["group"],
            meters_per_unit=d.get("meters_per_unit", METERS_PER_UNIT),
            rooms=tuple(
                Room(r["type"], tuple((p[0], p[1]) for p in r["polygon"])) for r in d["rooms"]
            ),
            doors=tuple(
                Door(
                    (d_["rooms"][0], d_["rooms"][1]),
                    (
                        (d_["segment"][0][0], d_["segment"][0][1]),
                        (d_["segment"][1][0], d_["segment"][1][1]),
                    ),
                )
                for d_ in d.get("doors", [])
            ),
            open_edges=tuple((e[0], e[1]) for e in d.get("open_edges", [])),
            meta=d.get("meta", {}),
        )


def validate_plan(plan: Plan, tol: float = 1e-6) -> list[str]:
    """Return the list of problems found in ``plan`` (empty when the plan is valid).

    Checks: known room types, simple polygons inside the grid, no overlap between rooms,
    doors lying on the walls of the rooms they connect, one entrance, connected room graph.
    """
    problems: list[str] = []
    shapes: list[Polygon] = []
    for i, room in enumerate(plan.rooms):
        if room.type not in ROOM_TYPES:
            problems.append(f"room {i}: unknown type {room.type!r}")
        shape = room.shape() if len(room.polygon) >= 3 else Polygon()
        shapes.append(shape)
        if shape.is_empty or not shape.is_valid or shape.area <= tol:
            problems.append(f"room {i}: invalid polygon")
            continue
        if not shape.exterior.is_ccw:
            problems.append(f"room {i}: polygon is not counter-clockwise")
        xs, ys = zip(*room.polygon, strict=True)
        if min(xs) < 0 or min(ys) < 0 or max(xs) > GRID_SIZE - 1 or max(ys) > GRID_SIZE - 1:
            problems.append(f"room {i}: outside the grid")
    if any(p.endswith("invalid polygon") for p in problems):
        return problems
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i].intersection(shapes[j]).area > 1e-3:
                problems.append(f"rooms {i} and {j} overlap")
    n = len(plan.rooms)
    for k, door in enumerate(plan.doors):
        i, j = door.rooms
        if not (0 <= i < n) or not (j == OUTSIDE or 0 <= j < n) or i == j:
            problems.append(f"door {k}: bad room indices {door.rooms}")
            continue
        for r in (i, j):
            if r != OUTSIDE and any(
                shapes[r].exterior.distance(ShapelyPoint(p)) > 1e-3 for p in door.segment
            ):
                problems.append(f"door {k}: not on a wall of room {r}")
    for i, j in plan.open_edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            problems.append(f"open edge {(i, j)}: bad room indices")
    if not plan.entrance_rooms():
        problems.append("no entrance door")
    g = plan.graph()
    if n and not nx.is_connected(g):
        problems.append("room graph is not connected")
    return problems


def write_plans(path: str | Path, plans: Iterable[Plan]) -> int:
    """Write plans as JSON Lines (gzip when the path ends with ``.gz``); return the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as f:
        for plan in plans:
            f.write(json.dumps(plan.to_dict(), separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_plans(path: str | Path) -> Iterator[Plan]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Plan.from_dict(json.loads(line))


def read_plans(path: str | Path) -> list[Plan]:
    return list(iter_plans(path))
