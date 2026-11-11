"""Headless rendering of 3D houses and gallery figures.

A small software renderer written with numpy: orthographic camera, z-buffer, flat shading
with one sun and a sky term, shadow map, outlines along creases, alpha-blended glass. It
needs no GPU, no OpenGL context and no display, so the same code runs on Windows, Kaggle,
CI and a CPU-only Space. Blender is not required.

Usage (``make render``): ``python -m housegen.render --config configs/data/swiss_dwellings.yaml``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen.geometry3d.export import House, build_house, load_building_config
from housegen.viz import draw_plan

BACKGROUND = np.array([1.0, 1.0, 1.0])
GROUND = np.array([0.86, 0.87, 0.84])
LINE = np.array([0.16, 0.16, 0.18])
WALL_CUT = np.array([0.24, 0.24, 0.26])


def _basis(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rows: right, up, forward (into the screen) for a camera looking at the origin."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    forward = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    if abs(forward[2]) > 0.999:  # looking straight down: north is up
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), forward])


def _rasterize(tris: np.ndarray, width: int, height: int, test: np.ndarray | None = None):
    """Z-buffer rasterisation of screen-space triangles (n, 3, 3): x, y in pixels, z depth.

    Returns ``(face index per pixel, depth per pixel)``. With ``test`` (an existing depth
    buffer), triangles are depth-tested against it without updating it and the result holds
    the nearest of them per pixel.
    """
    depth = np.full((height, width), np.inf)
    index = np.full((height, width), -1, dtype=np.int32)
    for k, tri in enumerate(tris):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xa, xb = (
            max(math.floor(min(x0, x1, x2)), 0),
            min(math.ceil(max(x0, x1, x2)), width),
        )
        ya, yb = (
            max(math.floor(min(y0, y1, y2)), 0),
            min(math.ceil(max(y0, y1, y2)), height),
        )
        if xa >= xb or ya >= yb:
            continue
        px = np.arange(xa, xb) + 0.5
        py = (np.arange(ya, yb) + 0.5)[:, None]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        z = w0 * z0 + w1 * z1 + w2 * z2
        window = depth[ya:yb, xa:xb]
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z < window)
        if test is not None:
            inside &= z < test[ya:yb, xa:xb]
        window[inside] = z[inside]
        index[ya:yb, xa:xb][inside] = k
    return index, depth


def _triangles(house: House):
    """Opaque and transparent triangles of a house with their RGBA colours and plane ids."""
    tris, colors, groups, walls = [], [], [], []
    for number, part in enumerate(house.parts):
        for vertices, faces in part.bodies:
            tris.append(vertices[faces])
            colors.append(np.tile(part.color, (len(faces), 1)))
            groups.append(np.full(len(faces), number))
            walls.append(np.full(len(faces), part.name == "walls"))
    tris, colors = np.concatenate(tris), np.concatenate(colors)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    # Faces of one part lying in one plane share an id: outlines are drawn between ids.
    offset = np.einsum("ij,ij->i", normals, tris[:, 0])
    key = np.column_stack([np.round(normals * 50), np.round(offset * 20), np.concatenate(groups)])
    _, plane = np.unique(key, axis=0, return_inverse=True)
    return tris, colors, normals, plane.reshape(-1), np.concatenate(walls)


def render_house(
    house: House,
    view: str = "iso",
    size: tuple[int, int] = (640, 480),
    azimuth: float | None = None,
    elevation: float = 28.0,
    supersample: int = 3,
    shadows: bool = True,
) -> np.ndarray:
    """RGB image (uint8, height x width x 3) of ``house``.

    ``view="iso"`` is a three-quarter view, turned towards the entrance door when
    ``azimuth`` is not given; ``view="top"`` looks straight down (use it on a house built
    with ``roof="none"`` to see the rooms).
    """
    tris, colors, normals, plane, is_wall = _triangles(house)
    if view == "top":
        azimuth, elevation, shadows = -90.0, 90.0, False  # shadows would hide the floors
        # Cut walls are drawn dark, as on an architect's plan.
        colors[is_wall & (normals[:, 2] > 0.9), :3] = WALL_CUT
    elif azimuth is None:
        azimuth = house.info.get("entrance_azimuth", -90.0) + 35.0
    cam = _basis(azimuth, elevation)
    sun = _basis(azimuth + 20.0, 45.0)  # lights both visible facades
    to_sun = -sun[2]

    points = tris.reshape(-1, 3)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, radius = (lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0
    ground_z = lo[2] + 0.12
    g = 3.0 * radius
    ground = np.array(
        [[[-g, -g, 0], [g, -g, 0], [g, g, 0]], [[-g, -g, 0], [g, g, 0], [-g, g, 0]]], dtype=float
    ) + np.array([center[0], center[1], ground_z])

    width, height = size[0] * supersample, size[1] * supersample
    screen = (points - center) @ cam.T
    span = np.abs(screen[:, :2]).max(axis=0)
    scale = 0.9 * min(width / (2 * span[0]), height / (2 * span[1]))

    def project(t: np.ndarray) -> np.ndarray:
        s = (t.reshape(-1, 3) - center) @ cam.T
        out = np.column_stack([width / 2 + s[:, 0] * scale, height / 2 - s[:, 1] * scale, s[:, 2]])
        return out.reshape(-1, 3, 3)

    opaque = colors[:, 3] >= 1.0
    facing = normals @ cam[2] < 0  # back-face culling: bodies are closed solids
    solid = np.flatnonzero(opaque & facing)
    all_tris = np.concatenate([tris[solid], ground])
    all_normals = np.concatenate([normals[solid], [[0, 0, 1.0]] * 2])
    all_colors = np.concatenate([colors[solid, :3], [GROUND] * 2])
    all_planes = np.concatenate([plane[solid], [-1, -1]])
    index, depth = _rasterize(project(all_tris), width, height)
    hit = index >= 0

    image = np.tile(BACKGROUND, (height, width, 1))
    face = index[hit]
    normal = all_normals[face]
    ys, xs = np.nonzero(hit)
    world = (
        center
        + ((xs + 0.5 - width / 2) / scale)[:, None] * cam[0]
        + ((height / 2 - ys - 0.5) / scale)[:, None] * cam[1]
        + depth[hit][:, None] * cam[2]
    )
    lit = np.clip(normal @ to_sun, 0.0, None)
    if shadows:
        res = 1024
        casters = tris[opaque]
        s_all = (casters.reshape(-1, 3) - center) @ sun.T
        s_span = np.abs(s_all[:, :2]).max(axis=0) * 1.05
        s_scale = res / (2 * s_span)

        def to_light(p: np.ndarray) -> np.ndarray:
            s = (p - center) @ sun.T
            return np.column_stack(
                [res / 2 + s[:, 0] * s_scale[0], res / 2 - s[:, 1] * s_scale[1], s[:, 2]]
            )

        _, light_depth = _rasterize(to_light(casters.reshape(-1, 3)).reshape(-1, 3, 3), res, res)
        probe = to_light(world + 0.04 * normal + 0.02 * to_sun)
        shade = np.zeros(len(probe))
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            u = np.clip((probe[:, 0] + dx * 1.2).astype(int), 0, res - 1)
            v = np.clip((probe[:, 1] + dy * 1.2).astype(int), 0, res - 1)
            shade += probe[:, 2] > light_depth[v, u] + 0.03
        lit = lit * (1.0 - shade / 5.0)
    sky = 0.5 + 0.5 * normal[:, 2]
    shading = 0.50 + 0.14 * sky + 0.48 * lit
    rgb = all_colors[face] * shading[:, None]
    is_ground = all_planes[face] < 0
    if is_ground.any():
        # The ground fades into the background away from the house.
        dist = np.linalg.norm(world[is_ground, :2] - center[:2], axis=1) / radius
        fade = np.clip((dist - 1.1) / 0.9, 0.0, 1.0)[:, None]
        rgb[is_ground] = rgb[is_ground] * (1 - fade) + BACKGROUND * fade
    image[hit] = np.clip(rgb, 0.0, 1.0)

    # Outlines where two different planes (or parts) meet, or at depth jumps.
    ids = np.where(hit, all_planes[index], -2)
    edge = np.zeros((height, width), dtype=bool)
    edge[:, :-1] |= (ids[:, :-1] != ids[:, 1:]) | (np.abs(depth[:, :-1] - depth[:, 1:]) > 0.25)
    edge[:-1, :] |= (ids[:-1, :] != ids[1:, :]) | (np.abs(depth[:-1, :] - depth[1:, :]) > 0.25)
    edge &= ids >= 0
    edge[:, 1:] |= edge[:, :-1] & (ids[:, 1:] >= 0)  # two supersampled pixels wide
    image[edge] = image[edge] * 0.45 + LINE * 0.55

    glass = np.flatnonzero(~opaque & facing)
    if len(glass):
        order = glass[np.argsort(-(tris[glass].mean(axis=1) - center) @ cam[2])]
        g_index, _ = _rasterize(project(tris[order]), width, height, test=depth)
        g_hit = g_index >= 0
        tint = colors[order][g_index[g_hit]]
        alpha = tint[:, 3:4]
        image[g_hit] = image[g_hit] * (1 - alpha) + tint[:, :3] * alpha

    image = image.reshape(size[1], supersample, size[0], supersample, 3).mean(axis=(1, 3))
    return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)


def render_triptych(plan, cfg: dict, roof: str = "hip", seed: int = 0, size=(560, 420)):
    """The three gallery views of one plan: (house with roof, cutaway from above, info)."""
    house = build_house(plan, cfg, roof=roof, seed=seed)
    cutaway = build_house(plan, cfg, roof="none", seed=seed)
    return render_house(house, "iso", size), render_house(cutaway, "top", size), house.info


def gallery(
    plans, cfg: dict, path: str | Path, roofs, seed: int = 0, per_row: int = 2
) -> list[dict]:
    """Gallery figure: for each plan, 3/4 view with roof, top view without roof, 2D plan."""
    rows = -(-len(plans) // per_row)
    fig = Figure(figsize=(per_row * 3 * 2.6, rows * 2.25), dpi=110)
    FigureCanvasAgg(fig)
    axes = np.atleast_2d(fig.subplots(rows, per_row * 3)).reshape(rows, per_row * 3)
    for ax in axes.ravel():
        ax.axis("off")
    infos = []
    for k, plan in enumerate(plans):
        iso, top, info = render_triptych(plan, cfg, roofs[k % len(roofs)], seed)
        infos.append(info | {"plan_id": plan.id})
        row, col = divmod(k, per_row)
        a_iso, a_top, a_plan = axes[row, col * 3 : col * 3 + 3]
        a_iso.imshow(iso)
        a_top.imshow(top)
        draw_plan(plan, a_plan, linewidth=1.0)
        a_iso.set_title(f"{plan.id}", fontsize=7)
        a_top.set_title(f"{info['roof_built']} roof, {info['windows']} windows", fontsize=7)
        a_plan.set_title(f"{len(plan.rooms)} rooms", fontsize=7)
    fig.tight_layout(pad=0.4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return infos


def main() -> None:
    from housegen.data.build import load_config
    from housegen.geometry3d.cli import sample_plans

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/data/<dataset>.yaml")
    parser.add_argument("--building", default="configs/building.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/figures/3d")
    args = parser.parse_args()
    cfg = load_building_config(args.building)
    plans = sample_plans(load_config(args.config), args.split, args.n, args.seed)
    roofs = ("hip", "hip", "gable", "flat")
    out = Path(args.out)
    gallery(plans, cfg, out / "gallery_real.png", roofs, args.seed)
    gallery(plans[:6], cfg, out / "gallery_real_small.png", roofs, args.seed)
    print(f"wrote {out / 'gallery_real.png'}")


if __name__ == "__main__":
    main()
