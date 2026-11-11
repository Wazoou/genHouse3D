"""Roofs: hip roof from the straight skeleton, gable roof, flat roof with parapet.

Straight skeleton without a library. No pure-Python straight-skeleton package is published
on PyPI and CGAL bindings could not be used on the development machine, so the hip roof is
built here, for *rectilinear* footprints only (which is what the plan format guarantees).

The straight skeleton is the trace left by the corners of a polygon whose edges all move
inwards at the same speed. For a rectilinear polygon that moving front is the mitred inward
offset, which shapely computes exactly. On a footprint snapped to a grid of step ``q``,
edges can only collapse or collide at offsets that are multiples of ``q / 2``. We therefore
walk through those offsets, detect the events by comparing the front we predict with the
real offset polygon, and emit one band of roof faces between two consecutive events. Each
band is a set of trapezoids, one per edge of the front, rising with the roof pitch.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from housegen.geometry3d.mesh import Body, boundary_edges, extrude, polygons, triangulate, weld


class RoofError(ValueError):
    """The requested roof cannot be built on this footprint."""


def _rings(geom) -> list[np.ndarray]:
    """Rings of a polygonal geometry, interior always on the left, collinear corners removed."""
    rings = []
    for poly in polygons(geom):
        poly = orient(poly.simplify(0), 1.0)
        if poly.area <= 1e-9:
            continue
        for ring in (poly.exterior, *poly.interiors):
            rings.append(np.asarray(ring.coords)[:-1])
    return rings


def _front(base: Polygon, offset: float) -> list[np.ndarray]:
    """Rings of ``base`` moved inwards by ``offset``, on the half-unit grid."""
    if offset == 0:
        return _rings(base)
    moved = base.buffer(-offset, join_style="mitre", mitre_limit=2.0)
    return _rings(shapely.set_precision(moved, 0.5))


def _trapezoids(rings: list[np.ndarray], rise: float) -> list[tuple[np.ndarray, ...]]:
    """One ``(a, b, a_top, b_top)`` per front edge after moving inwards by ``rise``.

    At a convex corner the two neighbouring edges cut the trapezoid at 45 degrees; at a
    reflex corner they open it at 45 degrees.
    """
    out = []
    for ring in rings:
        n = len(ring)
        d = np.roll(ring, -1, axis=0) - ring
        d = d / np.hypot(d[:, 0], d[:, 1])[:, None]
        normal = np.column_stack([-d[:, 1], d[:, 0]])  # points inwards (left of the edge)
        for k in range(n):
            a, b = ring[k], ring[(k + 1) % n]
            # The previous edge runs along -normal at a convex corner, +normal at a reflex one.
            start_convex = d[k - 1] @ normal[k] < 0
            end_convex = d[(k + 1) % n] @ normal[k] > 0
            a_top = a + rise * (normal[k] + (d[k] if start_convex else -d[k]))
            b_top = b + rise * (normal[k] + (-d[k] if end_convex else d[k]))
            out.append((a, b, a_top, b_top))
    return out


def _predicted_rings(rings: list[np.ndarray], traps: list[tuple]) -> list[np.ndarray]:
    """Rings the front would have if no event happened: the trapezoid tops, ring by ring."""
    out, k = [], 0
    for ring in rings:
        out.append(np.asarray([t[2] for t in traps[k : k + len(ring)]]))
        k += len(ring)
    return out


def _corner_triples(rings: list[np.ndarray]) -> set[tuple]:
    """Every corner with its two neighbours: equal sets mean equal rings, whatever the start.

    Comparing edges alone is not enough: when two reflex corners meet, the front splits in
    two without any edge changing.
    """
    triples = set()
    for ring in rings:
        pts = [(round(float(x), 3), round(float(y), 3)) for x, y in ring]
        n = len(pts)
        triples |= {(pts[k - 1], pts[k], pts[(k + 1) % n]) for k in range(n)}
    return triples


def _chain(points: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``points`` lying on segment ``p -> q`` (ends included), ordered along it."""
    length = float(np.hypot(*(q - p)))
    if length < 1e-9:
        return p[None, :]
    d = (q - p) / length
    rel = points - p
    along = rel @ d
    across = rel[:, 0] * d[1] - rel[:, 1] * d[0]
    on = (np.abs(across) < 1e-6) & (along > -1e-6) & (along < length + 1e-6)
    order = np.argsort(along[on])
    return points[on][order]


def _skeleton_surface(base: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Roof surface over ``base`` (integer rectilinear polygon): points (x, y, t) and faces.

    ``t`` is the distance travelled by the front, i.e. the roof height for a 45 degree pitch.
    """
    bands = []  # (start offset, end offset, trapezoids)
    level_points: dict[float, list[np.ndarray]] = {}
    start, rings = 0.0, _front(base, 0.0)
    limit = max(base.bounds[2] - base.bounds[0], base.bounds[3] - base.bounds[1])
    offset = 0.0
    while rings:
        offset += 0.5
        if offset > limit:
            raise RoofError("straight skeleton did not terminate")
        traps = _trapezoids(rings, offset - start)
        nxt = _front(base, offset)
        if _corner_triples(_predicted_rings(rings, traps)) == _corner_triples(nxt):
            continue  # no event: the same band keeps rising
        bands.append((start, offset, traps))
        level_points.setdefault(start, []).extend(p for t in traps for p in t[:2])
        level_points.setdefault(offset, []).extend(p for t in traps for p in t[2:])
        level_points[offset].extend(p for r in nxt for p in r)
        start, rings = offset, nxt
    points, faces = [], []

    def add(p, t) -> int:
        points.append((p[0], p[1], t))
        return len(points) - 1

    levels = {t: np.unique(np.round(np.asarray(p), 3), axis=0) for t, p in level_points.items()}
    for t0, t1, traps in bands:
        for a, b, a_top, b_top in traps:
            bottom = [add(p, t0) for p in _chain(levels[t0], a, b)]
            top = [add(p, t1) for p in _chain(levels[t1], a_top, b_top)]
            d = b - a
            i = j = 0
            # Zip the two chains into triangles (counter-clockwise seen from above).
            while i < len(bottom) - 1 or j < len(top) - 1:
                bottom_next = i < len(bottom) - 1 and (
                    j == len(top) - 1
                    or np.asarray(points[bottom[i + 1]][:2]) @ d
                    <= np.asarray(points[top[j + 1]][:2]) @ d
                )
                if bottom_next:
                    faces.append((bottom[i], bottom[i + 1], top[j]))
                    i += 1
                else:
                    faces.append((bottom[i], top[j + 1], top[j]))
                    j += 1
    return np.asarray(points, dtype=float), np.asarray(faces, dtype=int)


def _snap(footprint: Polygon, quantum: float) -> Polygon:
    """Footprint in units of ``quantum``, on the integer grid, holes filled."""
    scaled = affinity.scale(footprint, 1.0 / quantum, 1.0 / quantum, origin=(0, 0))
    snapped = shapely.set_precision(Polygon(scaled.exterior), 1.0)
    parts = polygons(snapped)
    if len(parts) != 1:
        raise RoofError("footprint is not a single polygon")
    base = orient(Polygon(parts[0].exterior).simplify(0), 1.0)
    c = np.asarray(base.exterior.coords)
    d = c[1:] - c[:-1]
    if np.any((np.abs(d[:, 0]) > 1e-9) & (np.abs(d[:, 1]) > 1e-9)):
        raise RoofError("footprint is not rectilinear")
    return base


def hip_roof(
    footprint: Polygon,
    z: float,
    pitch_deg: float = 30.0,
    overhang: float = 0.3,
    fascia: float = 0.15,
    quantum: float = 0.1,
) -> Body:
    """Closed hip roof solid over a rectilinear ``footprint`` (metres), resting at height ``z``.

    The solid has a flat underside at ``z`` (ceiling and soffit), a vertical ``fascia`` along
    the eaves and the straight-skeleton surface on top.
    """
    base = _snap(footprint.buffer(overhang, join_style="mitre"), quantum)
    pts, faces = _skeleton_surface(base)
    tan = math.tan(math.radians(pitch_deg))
    top = np.column_stack([pts[:, :2] * quantum, z + fascia + pts[:, 2] * quantum * tan])
    cap_pts, cap_tris = triangulate(base)
    n = len(top)
    cap = np.column_stack([cap_pts * quantum, np.full(len(cap_pts), z)])
    rim = np.column_stack([cap_pts * quantum, np.full(len(cap_pts), z + fascia)])
    edges = boundary_edges(cap_tris)
    a, b = edges[:, 0] + n, edges[:, 1] + n
    m = len(cap)
    sides = np.concatenate([np.column_stack([a, b, b + m]), np.column_stack([a, b + m, a + m])])
    body = weld(
        np.concatenate([top, cap, rim]), np.concatenate([faces, cap_tris[:, ::-1] + n, sides])
    )
    return body


def is_nearly_rectangular(footprint: Polygon, min_fill: float = 0.85) -> bool:
    return footprint.area / box(*footprint.bounds).area >= min_fill


def gable_roof(
    footprint: Polygon,
    z: float,
    pitch_deg: float = 30.0,
    overhang: float = 0.3,
    thickness: float = 0.15,
) -> tuple[Body, Body]:
    """Gable roof over the bounding box of ``footprint``: (roof slab, gable wall solid).

    The ridge follows the long side. The gable solid is the triangular prism closing the
    attic between the two end walls; the roof is a chevron-shaped slab laid on it, extended
    by the overhang on the four sides.
    """
    x0, y0, x1, y1 = footprint.bounds
    along_x = (x1 - x0) >= (y1 - y0)
    span, length = ((y1 - y0), (x1 - x0)) if along_x else ((x1 - x0), (y1 - y0))
    tan = math.tan(math.radians(pitch_deg))
    half = span / 2.0
    # Cross-sections in (u, height) with u across the ridge, extruded along the ridge.
    attic = Polygon([(-half, 0.0), (half, 0.0), (0.0, half * tan)])
    w = half + overhang
    drop = overhang * tan
    lift = thickness / math.cos(math.radians(pitch_deg))
    chevron = Polygon(
        [(-w, -drop), (0.0, half * tan), (w, -drop), (w, -drop + lift),
         (0.0, half * tan + lift), (-w, -drop + lift)]
    )  # fmt: skip
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    def place(section: Polygon, v0: float, v1: float) -> Body:
        vertices, faces = extrude(section, v0, v1)  # columns: u, height, v
        u, h, v = vertices.T
        # Both mappings are rotations of the (u, height, v) frame: the winding is kept.
        if along_x:
            return np.column_stack([center[0] + v, center[1] + u, z + h]), faces
        return np.column_stack([center[0] - u, center[1] + v, z + h]), faces

    roof = place(chevron, -length / 2.0 - overhang, length / 2.0 + overhang)
    gable = place(attic, -length / 2.0, length / 2.0)
    return roof, gable


def flat_roof(
    footprint: Polygon,
    z: float,
    thickness: float = 0.25,
    parapet_height: float = 0.4,
    parapet_thickness: float = 0.2,
) -> list[Body]:
    """Flat roof slab over ``footprint`` with a parapet around it."""
    slab = extrude(footprint, z, z + thickness)
    inner = footprint.buffer(-parapet_thickness, join_style="mitre")
    ring = footprint.difference(inner) if not inner.is_empty else footprint
    bodies = [slab]
    for part in polygons(ring):
        bodies.append(extrude(part, z + thickness, z + thickness + parapet_height))
    return bodies
