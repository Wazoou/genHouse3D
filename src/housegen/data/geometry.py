"""Geometry helpers turning raw room outlines into a clean rectilinear partition.

Raw datasets describe rooms by their net inner outline: neighbouring rooms are separated by
the thickness of the wall between them and the building is rarely aligned with the axes. The
unified format wants rooms that share edges on an axis-aligned grid, so we

1. rotate the dwelling by its dominant wall direction (:func:`dominant_angle`),
2. turn every nearly rectilinear outline into an exactly rectilinear ring (:func:`rectify_ring`),
3. move facing wall faces onto the wall centre line (:func:`snap_walls`),
4. round to the grid and drop degenerate corners (:func:`clean_ring`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge


def dominant_angle(polygons: Sequence[Polygon]) -> float:
    """Dominant wall direction modulo 90 degrees, in radians within (-pi/4, pi/4].

    Length-weighted circular mean of the edge directions multiplied by 4, so that the four
    directions of a rectilinear layout vote for the same angle.
    """
    acc = np.zeros(2)
    for poly in polygons:
        c = np.asarray(poly.exterior.coords)
        d = c[1:] - c[:-1]
        length = np.hypot(d[:, 0], d[:, 1])
        a = np.arctan2(d[:, 1], d[:, 0]) * 4.0
        acc += [np.sum(length * np.cos(a)), np.sum(length * np.sin(a))]
    return math.atan2(acc[1], acc[0]) / 4.0


def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def clean_ring(ring: np.ndarray) -> np.ndarray | None:
    """Remove repeated and collinear corners (spikes included); return a CCW ring or ``None``."""
    pts = [tuple(p) for p in np.asarray(ring).tolist()]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        out: list[tuple] = []
        n = len(pts)
        for k in range(n):
            prev, cur, nxt = pts[k - 1], pts[k], pts[(k + 1) % n]
            cross = (cur[0] - prev[0]) * (nxt[1] - cur[1]) - (cur[1] - prev[1]) * (nxt[0] - cur[0])
            if cur == prev or cross == 0:
                changed = True
                # Drop one corner per pass around a degenerate spot so neighbours are re-tested.
                out.extend(pts[k + 1 :])
                break
            out.append(cur)
        pts = out
    if len(pts) < 4:
        return None
    arr = np.asarray(pts)
    area = signed_area(arr)
    if area == 0:
        return None
    return arr if area > 0 else arr[::-1].copy()


def rectify_ring(
    polygon: Polygon,
    angle_tol_deg: float = 12.0,
    max_diagonal: float = 0.15,
    simplify_tol: float = 0.03,
    max_area_change: float = 0.05,
) -> np.ndarray | None:
    """Exactly rectilinear CCW ring approximating a nearly rectilinear ``polygon``.

    Every edge is classified as horizontal or vertical; consecutive edges of the same class
    are merged into one wall face placed at their length-weighted mean coordinate. Returns
    ``None`` when the outline is not rectilinear: an edge longer than ``max_diagonal`` metres
    deviating from both axes by more than ``angle_tol_deg``, or an area change above
    ``max_area_change``. Holes are ignored.
    """
    simple = Polygon(polygon.exterior).simplify(simplify_tol, preserve_topology=True)
    if simple.is_empty or not simple.is_valid or simple.geom_type != "Polygon":
        return None
    c = np.asarray(simple.exterior.coords)[:-1]
    if signed_area(c) < 0:
        c = c[::-1]
    d = np.roll(c, -1, axis=0) - c
    length = np.hypot(d[:, 0], d[:, 1])
    keep = length > 1e-9
    c, d, length = c[keep], d[keep], length[keep]
    if len(c) < 4:
        return None
    horizontal = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    deviation = np.degrees(
        np.arctan2(
            np.minimum(np.abs(d[:, 0]), np.abs(d[:, 1])),
            np.maximum(np.abs(d[:, 0]), np.abs(d[:, 1])),
        )
    )
    if np.any((deviation > angle_tol_deg) & (length > max_diagonal)):
        return None
    # Start on a class change so that runs do not wrap around the end of the array.
    changes = np.flatnonzero(horizontal != np.roll(horizontal, 1))
    if len(changes) < 4:
        return None
    start = int(changes[0])
    order = np.roll(np.arange(len(c)), -start)
    runs: list[tuple[bool, float]] = []  # (is_horizontal, coordinate of the wall face)
    k = 0
    mid = (c + np.roll(c, -1, axis=0)) / 2.0
    while k < len(order):
        cls = bool(horizontal[order[k]])
        idx = []
        while k < len(order) and bool(horizontal[order[k]]) == cls:
            idx.append(order[k])
            k += 1
        idx = np.asarray(idx)
        weights = np.abs(d[idx, 0] if cls else d[idx, 1])
        values = mid[idx, 1] if cls else mid[idx, 0]
        runs.append((cls, float(np.average(values, weights=weights))))
    if len(runs) < 4 or len(runs) % 2:
        return None
    ring = []
    for k, (cls, value) in enumerate(runs):
        nxt_value = runs[(k + 1) % len(runs)][1]
        # The corner between run k and run k+1 takes x from the vertical one, y from the other.
        ring.append((nxt_value, value) if cls else (value, nxt_value))
    ring = np.asarray(ring)
    rect = Polygon(ring)
    if not rect.is_valid or rect.area <= 0:
        return None
    if abs(rect.area - polygon.area) > max_area_change * polygon.area:
        return None
    if signed_area(ring) < 0:
        ring = ring[::-1].copy()
    return ring


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def snap_walls(
    rings: Sequence[np.ndarray],
    max_wall: float = 0.7,
    min_overlap: float = 0.1,
    align_tol: float = 0.1,
    max_overlap: float = 1e-6,
    block_share: float = 0.0,
) -> list[np.ndarray]:
    """Close the wall gaps between rectilinear rings so that neighbours share edges.

    A wall is a pair of faces looking at each other (the east face of one room and the west
    face of the next, or north / south), at most ``max_wall`` metres apart, overlapping by at
    least ``min_overlap`` metres, with nothing but wall in between. Faces linked this way are
    moved to the middle of their group, i.e. onto the wall centre line. Faces without a
    partner (outer walls) do not move. Finally, faces within ``align_tol`` metres of each
    other along an axis are put on a common line. With ``max_overlap`` > 0, faces that went
    past each other by at most that much (rooms overlapping slightly) are paired too.
    Two faces are not paired when other rooms cover more than ``block_share`` of the strip
    between them (0: any contact blocks, as for real plans where a thin room may sit there).
    """
    polys = [Polygon(r) for r in rings]
    # One record per edge: ring, index, axis (0: vertical face, 1: horizontal face),
    # coordinate, extent, and whether the room lies on the low-coordinate side of the face.
    recs = []
    for r, ring in enumerate(rings):
        nxt = np.roll(ring, -1, axis=0)
        for k in range(len(ring)):
            (x0, y0), (x1, y1) = ring[k], nxt[k]
            if x0 == x1:  # vertical edge; CCW ring: going up means the room is to the west
                recs.append((r, k, 0, x0, min(y0, y1), max(y0, y1), y1 > y0))
            else:  # horizontal edge; going left means the room is below
                recs.append((r, k, 1, y0, min(x0, x1), max(x0, x1), x1 < x0))
    uf = _UnionFind(len(recs))
    arr = np.array([(a[2], a[3], a[4], a[5], a[6]) for a in recs], dtype=float)
    for axis in (0, 1):
        low = np.flatnonzero((arr[:, 0] == axis) & (arr[:, 4] == 1))  # room on the low side
        high = np.flatnonzero((arr[:, 0] == axis) & (arr[:, 4] == 0))
        if not len(low) or not len(high):
            continue
        gap = arr[high, 1][None, :] - arr[low, 1][:, None]
        lo = np.maximum(arr[low, 2][:, None], arr[high, 2][None, :])
        hi = np.minimum(arr[low, 3][:, None], arr[high, 3][None, :])
        cand = (gap >= -max_overlap) & (gap <= max_wall) & (hi - lo >= min_overlap)
        for a, b in zip(*np.nonzero(cand), strict=True):
            ia, ib = int(low[a]), int(high[b])
            if recs[ia][0] == recs[ib][0] and recs[ia][1] == recs[ib][1]:
                continue
            if gap[a, b] > 0.02:
                eps = 0.005
                c0, c1 = arr[ia, 1] + eps, arr[ib, 1] - eps
                e0, e1 = lo[a, b] + eps, hi[a, b] - eps
                strip = box(c0, e0, c1, e1) if axis == 0 else box(e0, c0, e1, c1)
                others = [p for k, p in enumerate(polys) if k not in (recs[ia][0], recs[ib][0])]
                if block_share <= 0:
                    if any(p.intersects(strip) for p in polys):
                        continue
                elif sum(p.intersection(strip).area for p in others) > block_share * strip.area:
                    continue
            uf.union(ia, ib)
    groups: dict[int, list[int]] = {}
    for i in range(len(recs)):
        groups.setdefault(uf.find(i), []).append(i)
    coord = np.empty(len(recs))
    for members in groups.values():
        values = arr[members, 1]
        coord[members] = (values.min() + values.max()) / 2.0
    # Faces that are almost in line (two walls a few centimetres apart) are put on one line,
    # otherwise rounding to the grid leaves one-unit jogs, slivers and tiny overlaps.
    extent = arr[:, 3] - arr[:, 2]
    for axis in (0, 1):
        idx = np.flatnonzero(arr[:, 0] == axis)
        idx = idx[np.argsort(coord[idx], kind="stable")]
        start = 0
        for stop in range(1, len(idx) + 1):
            if stop == len(idx) or coord[idx[stop]] - coord[idx[start]] > align_tol:
                group = idx[start:stop]
                coord[group] = np.average(coord[group], weights=extent[group] + 1e-9)
                start = stop
    out = [np.array(r, dtype=float) for r in rings]
    for i, (r, k, axis, *_rest) in enumerate(recs):
        # Edge k joins corners k and k+1: both take the snapped coordinate of this face.
        n = len(out[r])
        out[r][k, axis] = coord[i]
        out[r][(k + 1) % n, axis] = coord[i]
    return out


def straight_pieces(geom) -> list[LineString]:
    """Split any linear geometry into maximal straight two-point segments."""
    lines = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "LineString"]
    if not lines:
        return []
    merged = linemerge(lines) if len(lines) > 1 else lines[0]
    pieces: list[LineString] = []
    for line in getattr(merged, "geoms", [merged]):
        coords = list(line.simplify(0).coords)
        pieces.extend(LineString(coords[k : k + 2]) for k in range(len(coords) - 1))
    return pieces


def place_on_wall(
    pieces: list[LineString], center: Point, width: float, max_shift: float, min_wall: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Footprint of an opening of ``width`` centred as close as possible to ``center``.

    The opening is laid on the nearest straight wall piece that is at least ``min_wall`` long
    and at most ``max_shift`` away; it is shortened or slid to stay within the piece.
    """
    best = None
    for piece in pieces:
        if piece.length >= min_wall:
            dist = piece.distance(center)
            if dist <= max_shift and (best is None or dist < best[0]):
                best = (dist, piece)
    if best is None:
        return None
    piece = best[1]
    half = min(width, piece.length) / 2.0
    t = min(max(piece.project(center), half), piece.length - half)
    a, b = piece.interpolate(t - half), piece.interpolate(t + half)
    return (round(a.x, 2), round(a.y, 2)), (round(b.x, 2), round(b.y, 2))
