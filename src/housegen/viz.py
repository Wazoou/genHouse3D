"""2D rendering of floorplans: exploration figures, galleries and fixed-style rasters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as PolygonPatch

from housegen.data.schema import GRID_SIZE, ROOM_TYPES, Plan

ROOM_COLORS: dict[str, str] = {
    "living": "#f2c879",
    "kitchen": "#e58f65",
    "dining": "#f0a8a0",
    "bedroom": "#8fb8de",
    "bathroom": "#7fd1c7",
    "corridor": "#d9d4c7",
    "storage": "#b7a6c9",
    "balcony": "#a9d18e",
}
WALL_COLOR = "#2b2b2b"
DOOR_COLOR = "#ffffff"
ENTRANCE_COLOR = "#d62828"


def draw_plan(plan: Plan, ax, doors: bool = True, margin: float = 6.0, linewidth: float = 1.6):
    """Draw ``plan`` on a matplotlib axis: one colour per room type, doors as wall openings."""
    drawable = [r for r in plan.rooms if len(r.polygon) >= 3]  # collapsed rooms are skipped
    for room in drawable:
        ax.add_patch(
            PolygonPatch(
                np.asarray(room.polygon),
                closed=True,
                facecolor=ROOM_COLORS.get(room.type, "#cccccc"),
                edgecolor=WALL_COLOR,
                linewidth=linewidth,
                joinstyle="miter",
            )
        )
    if doors:
        for door in plan.doors:
            (x0, y0), (x1, y1) = door.segment
            color = ENTRANCE_COLOR if door.is_entrance else DOOR_COLOR
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=linewidth * 1.5,
                solid_capstyle="butt",
                zorder=3,
            )
    if not drawable:
        ax.axis("off")
        return
    pts = np.concatenate([np.asarray(r.polygon, dtype=float) for r in drawable])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    half = max(x1 - x0, y1 - y0) / 2 + margin
    ax.set_xlim((x0 + x1) / 2 - half, (x0 + x1) / 2 + half)
    ax.set_ylim((y0 + y1) / 2 - half, (y0 + y1) / 2 + half)
    ax.set_aspect("equal")
    ax.axis("off")


def legend_handles(types: Sequence[str] = ROOM_TYPES) -> list[Patch]:
    handles = [Patch(facecolor=ROOM_COLORS[t], edgecolor=WALL_COLOR, label=t) for t in types]
    handles.append(Patch(facecolor=ENTRANCE_COLOR, edgecolor=ENTRANCE_COLOR, label="entrance door"))
    return handles


def plot_plans(
    plans: Sequence[Plan],
    path: str | Path | None = None,
    ncols: int = 5,
    titles: Sequence[str] | None = None,
    cell: float = 2.6,
    dpi: int = 130,
) -> Figure:
    """Grid of plans with a shared legend; saved to ``path`` when given."""
    nrows = -(-len(plans) // ncols)
    fig = Figure(figsize=(ncols * cell, nrows * cell + 0.6), dpi=dpi)
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for ax in axes:
        ax.axis("off")
    for k, plan in enumerate(plans):
        draw_plan(plan, axes[k])
        if titles is not None:
            axes[k].set_title(titles[k], fontsize=7)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncols=len(ROOM_TYPES) + 1,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.6 / (nrows * cell + 0.6), 1, 1))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def render_plan_image(plan: Plan, size: int = 256, doors: bool = True) -> np.ndarray:
    """Fixed-style RGB raster (``size`` x ``size`` uint8) of the full grid canvas.

    The same function renders real and generated plans, which keeps image-based metrics
    such as the FID comparable.
    """
    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    draw_plan(plan, ax, doors=doors, linewidth=1.0)
    ax.set_xlim(0, GRID_SIZE - 1)
    ax.set_ylim(0, GRID_SIZE - 1)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
