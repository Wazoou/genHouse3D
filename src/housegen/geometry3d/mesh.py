"""Mesh primitives: watertight prisms from 2D polygons, mesh parts and their validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

Body = tuple[np.ndarray, np.ndarray]  # (vertices (n, 3) float, faces (m, 3) int)


def polygons(geom) -> list[Polygon]:
    """Non-empty polygons of any shapely geometry (lines and points are dropped)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return [p for g in getattr(geom, "geoms", []) for p in polygons(g)]


def triangulate(polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation: (points (n, 2), CCW triangles (m, 3)).

    Only the polygon's own vertices are used, so the boundary of the triangulation is
    exactly the boundary of the polygon.
    """
    polygon = orient(polygon.simplify(0), 1.0)
    index: dict[tuple[float, float], int] = {}
    tris = []
    for tri in shapely.constrained_delaunay_triangles(polygon).geoms:
        c = np.asarray(tri.exterior.coords)[:3]
        cross = (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[1, 1] - c[0, 1]) * (
            c[2, 0] - c[0, 0]
        )
        if abs(cross) < 1e-12:
            continue
        ids = [index.setdefault((round(x, 6), round(y, 6)), len(index)) for x, y in c]
        tris.append(ids if cross > 0 else ids[::-1])
    return _split_pinches(
        np.asarray(list(index), dtype=float), np.asarray(tris, dtype=int).reshape(-1, 3)
    )


def _split_pinches(pts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate the vertices where the polygon touches itself at a single point.

    Around such a vertex the triangles form several fans that only meet at the vertex; an
    extrusion would turn it into an edge shared by four faces. Each fan gets its own copy.
    """
    pts, tris = pts.tolist(), tris.copy()
    around: dict[int, list[int]] = {}
    for t, tri in enumerate(tris):
        for v in tri:
            around.setdefault(int(v), []).append(t)
    for v, incident in around.items():
        fan = {t: t for t in incident}  # union-find over the triangles around v

        def find(t: int, fan=fan) -> int:
            while fan[t] != t:
                t = fan[t]
            return t

        by_neighbour: dict[int, int] = {}
        for t in incident:
            for w in tris[t]:
                if w != v:
                    other = by_neighbour.setdefault(int(w), t)
                    fan[find(t)] = find(other)
        roots = sorted({find(t) for t in incident})
        for root in roots[1:]:
            pts.append(pts[v])
            for t in incident:
                if find(t) == root:
                    tris[t][tris[t] == v] = len(pts) - 1
    return np.asarray(pts, dtype=float), tris


def boundary_edges(tris: np.ndarray) -> np.ndarray:
    """Directed edges used by exactly one triangle (the interior lies on their left)."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    seen = {(int(a), int(b)) for a, b in edges}
    return np.asarray([e for e in edges if (int(e[1]), int(e[0])) not in seen], dtype=int)


def extrude(polygon: Polygon, z0: float, z1: float) -> Body:
    """Watertight prism over ``polygon`` (holes allowed) between ``z0`` and ``z1``."""
    pts, tris = triangulate(polygon)
    n = len(pts)
    vertices = np.concatenate(
        [np.column_stack([pts, np.full(n, z0)]), np.column_stack([pts, np.full(n, z1)])]
    )
    edges = boundary_edges(tris)
    a, b = edges[:, 0], edges[:, 1]
    sides = np.concatenate([np.column_stack([a, b, b + n]), np.column_stack([a, b + n, a + n])])
    return vertices, np.concatenate([tris[:, ::-1], tris + n, sides])


def extrude_all(geom, z0: float, z1: float) -> list[Body]:
    return [extrude(p, z0, z1) for p in polygons(geom) if p.area > 1e-8]


def weld(points: np.ndarray, faces: np.ndarray, decimals: int = 6) -> Body:
    """Merge vertices with equal rounded coordinates and drop collapsed faces."""
    keys = np.round(points, decimals)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return unique, faces[keep]


@dataclass
class Part:
    """Bodies sharing one material (walls, roof, glass...)."""

    name: str
    color: tuple[float, float, float, float]
    bodies: list[Body] = field(default_factory=list)

    def mesh(self) -> trimesh.Trimesh:
        """All bodies in one mesh (vertices are not merged across bodies)."""
        vertices, faces, offset = [], [], 0
        for v, f in self.bodies:
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)


def body_problems(body: Body, tol: float = 1e-9) -> list[str]:
    """Defects of one body: open edges, inconsistent winding, inverted or degenerate faces."""
    vertices, faces = body
    if not np.isfinite(vertices).all():
        return ["non-finite vertex"]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    problems = []
    if not mesh.is_watertight:
        problems.append("not watertight")
    elif not mesh.is_winding_consistent:
        problems.append("inconsistent winding")
    elif mesh.volume <= tol:
        problems.append("normals point inwards")
    if (mesh.area_faces <= tol).any():
        problems.append("degenerate face")
    return problems
