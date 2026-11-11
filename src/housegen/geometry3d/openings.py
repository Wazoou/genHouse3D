"""Doors and windows: where they go, and the holes they cut in the walls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from housegen.data.geometry import straight_pieces
from housegen.geometry3d.walls import Layout, fill_holes


@dataclass
class Opening:
    """A hole through a wall, along ``p0 -> p1`` on a room edge (grid units), from z0 to z1."""

    kind: str  # "door" | "entrance" | "passage" | "window"
    p0: np.ndarray
    p1: np.ndarray
    z0: float  # metres
    z1: float
    room: int
    outward: np.ndarray | None  # unit normal towards the outside, for outer walls

    @property
    def width(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    def _axis(self, layout: Layout) -> tuple[LineString, float]:
        """Centre line of the wall under the opening and the wall thickness."""
        if self.outward is None:
            return LineString([self.p0, self.p1]), layout.t_int
        shift = self.outward * layout.t_ext / 2.0
        return LineString([self.p0 + shift, self.p1 + shift]), layout.t_ext

    def cut(self, layout: Layout) -> Polygon:
        """Rectangle removed from the wall footprint (slightly deeper than the wall)."""
        axis, thickness = self._axis(layout)
        return axis.buffer(thickness / 2.0 + 0.1, cap_style="flat")

    def panel(self, layout: Layout, thickness: float) -> Polygon:
        """Footprint of the door leaf or the glass pane, centred in the wall."""
        axis, _ = self._axis(layout)
        return axis.buffer(thickness / layout.unit / 2.0, cap_style="flat")


def _outward(layout: Layout, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    d = (p1 - p0) / np.hypot(*(p1 - p0))
    n = np.array([-d[1], d[0]])
    probe = (p0 + p1) / 2.0 + 0.25 * n
    return -n if layout.interior.contains(Point(probe)) else n


def _centered(p0: np.ndarray, p1: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-segment of ``p0 -> p1`` of at most ``width``, around its middle."""
    length = np.hypot(*(p1 - p0))
    half = min(width, length) / 2.0
    mid, d = (p0 + p1) / 2.0, (p1 - p0) / length
    return mid - half * d, mid + half * d


def place_doors(layout: Layout, cfg: dict) -> list[Opening]:
    """Door openings of the plan, plus full-height passages onto open balconies."""
    unit, doors_cfg = layout.unit, cfg["doors"]
    openings: list[Opening] = []
    for door in layout.plan.doors:
        i, j = door.rooms
        p0, p1 = (layout.to_local(p) for p in door.segment)
        if door.is_entrance:
            if layout.is_outdoor(i):
                continue
            a, b = _centered(p0, p1, doors_cfg["entrance_width"] / unit)
            openings.append(
                Opening("entrance", a, b, 0.0, doors_cfg["height"], i, _outward(layout, a, b))
            )
            continue
        if layout.is_outdoor(i) and layout.is_outdoor(j):
            continue
        a, b = _centered(p0, p1, doors_cfg["width"] / unit)
        outdoor = layout.is_outdoor(i) or layout.is_outdoor(j)
        room = j if layout.is_outdoor(i) else i
        outward = _outward(layout, a, b) if outdoor else None
        openings.append(Opening("door", a, b, 0.0, doors_cfg["height"], room, outward))
    for i, j, kind in layout.plan.edges():
        if kind == "open" and layout.is_outdoor(i) != layout.is_outdoor(j):
            # A room open onto a balcony: leave a doorway in the outer wall between them.
            shared = layout.rooms[i].exterior.intersection(layout.rooms[j].exterior)
            pieces = sorted(straight_pieces(shared), key=lambda p: -p.length)
            if pieces and pieces[0].length > 2 * 0.2 / unit:
                c = np.asarray(pieces[0].coords)
                a, b = _centered(c[0], c[1], pieces[0].length - 2 * 0.2 / unit)
                room = j if layout.is_outdoor(i) else i
                openings.append(
                    Opening("passage", a, b, 0.0, doors_cfg["height"], room, _outward(layout, a, b))
                )
    return openings


def _free_intervals(length: float, blocked: list[tuple[float, float]], margin: float):
    """Parts of ``[margin, length - margin]`` left after removing ``blocked`` intervals."""
    free = [(margin, length - margin)]
    for lo, hi in blocked:
        free = [
            part
            for a, b in free
            for part in ((a, min(b, lo)), (max(a, hi), b))
            if part[1] - part[0] > 1e-9
        ]
    return free


def place_windows(
    layout: Layout, cfg: dict, doors: list[Opening], rng: np.random.Generator
) -> list[Opening]:
    """Windows on the outer walls, following the per-room-type rules of the config.

    For each room, the number of windows is drawn in the rule's range; each window is
    centred in the longest free stretch of facade left (clear of wall ends, doors and the
    windows already placed) and shrunk to fit; stretches too short for the rule's minimum
    width get none.
    """
    unit, win = layout.unit, cfg["windows"]
    margin = win["margin"] / unit
    max_top = cfg["walls"]["height"] - 0.15
    facade = fill_holes(layout.interior).boundary
    windows: list[Opening] = []
    for room, room_type in enumerate(layout.types):
        rule = win["rules"].get(room_type)
        if rule is None or layout.is_outdoor(room):
            continue
        count = int(rng.integers(rule["count"][0], rule["count"][1] + 1))
        width, min_width = rule["width"] / unit, rule["min_width"] / unit
        z0, z1 = rule["sill"], min(rule["sill"] + rule["height"], max_top)
        stretches = []  # (length, start point, unit direction)
        for piece in straight_pieces(layout.rooms[room].exterior.intersection(facade)):
            c = np.asarray(piece.coords)
            d = (c[1] - c[0]) / piece.length
            blocked = []
            for door in doors:
                if (
                    door.outward is not None
                    and piece.distance(Point((door.p0 + door.p1) / 2)) < 1e-6
                ):
                    s = sorted([float((door.p0 - c[0]) @ d), float((door.p1 - c[0]) @ d)])
                    blocked.append((s[0] - margin, s[1] + margin))
            for lo, hi in _free_intervals(piece.length, blocked, margin):
                stretches.append((hi - lo, c[0] + lo * d, d))
        for _ in range(count):
            if not stretches:
                break
            stretches.sort(key=lambda s: -s[0])
            length, start, d = stretches.pop(0)
            w = min(width, length)
            if w < min_width:
                break
            a = start + (length - w) / 2.0 * d
            b = a + w * d
            windows.append(Opening("window", a, b, z0, z1, room, _outward(layout, a, b)))
            side = (length - w) / 2.0 - margin
            if side > 0:
                stretches += [(side, start, d), (side, b + margin * d, d)]
    return windows
