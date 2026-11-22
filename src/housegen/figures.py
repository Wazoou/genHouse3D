"""README figures made from a trained model.

Usage: ``python -m housegen.figures --checkpoint runs/base/ema.pt``

- ``pipeline.png``: one house generated step by step (graph, noisy corners, denoising, raw
  plan, cleaned plan, 3D);
- ``hero.png``: the first six valid houses of a seeded random draw of test graphs, each with
  its room graph and plan (the full, unfiltered draw is the gallery of ``housegen.gallery``).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as PolygonPatch

from housegen.data.build import load_config, load_split
from housegen.geometry3d.export import build_house, load_building_config
from housegen.graphs import RoomGraph
from housegen.model.encoding import PlanTensors, to_grid
from housegen.postprocess import attach_doors, clean_plan, validity
from housegen.render import render_house
from housegen.sample import decode_rooms, load_checkpoint
from housegen.viz import ENTRANCE_COLOR, ROOM_COLORS, WALL_COLOR, draw_plan


def draw_graph(graph: RoomGraph, ax, seed: int = 0) -> None:
    """Bubble diagram: rooms coloured by type, doors as solid lines, openings dashed."""
    g = graph.to_networkx()
    pos = nx.kamada_kawai_layout(g) if len(g) > 1 else {0: np.zeros(2)}
    for i, j, kind in graph.edges:
        (x0, y0), (x1, y1) = pos[i], pos[j]
        ax.plot([x0, x1], [y0, y1], color=WALL_COLOR, linewidth=1.2,
                linestyle="-" if kind == "door" else (0, (2, 2)), zorder=1)  # fmt: skip
    for i, room_type in enumerate(graph.types):
        edge = ENTRANCE_COLOR if i in graph.entrances else WALL_COLOR
        ax.scatter(*pos[i], s=170, color=ROOM_COLORS[room_type], edgecolors=edge,
                   linewidths=2.0 if i in graph.entrances else 1.0, zorder=2)  # fmt: skip
    ax.set_aspect("equal")
    ax.margins(0.15)
    ax.axis("off")


def draw_corners(x: np.ndarray, graph: RoomGraph, ax, outline: bool = True) -> None:
    """Corners in model units, coloured by room, with the outline they would draw."""
    grid = to_grid(x)
    k = 0
    for room_type, n in zip(graph.types, graph.corners, strict=True):
        pts = grid[k : k + n]
        if outline:
            ax.add_patch(PolygonPatch(pts, closed=True, fill=False, linewidth=0.6,
                                      edgecolor=ROOM_COLORS[room_type]))  # fmt: skip
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=ROOM_COLORS[room_type], edgecolors=WALL_COLOR,
                   linewidths=0.3, zorder=2)  # fmt: skip
        k += n
    ax.set_xlim(-20, 275)
    ax.set_ylim(-20, 275)
    ax.set_aspect("equal")
    ax.axis("off")


def pipeline_figure(model, diffusion, cfg, graph, building, path, seed=0, device="cpu") -> bool:
    """One generation step by step; returns whether the cleaned plan is valid."""
    data = PlanTensors.from_graphs([graph], device)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = data.batch(torch.arange(1), generator, room_ids=cfg["augment"]["room_ids"])
    keep = {999: None, 600: None, 300: None, 100: None}
    states = {}

    def record(i, x, x0):
        for t in keep:
            if t not in states and i <= t:
                states[t] = x[0].cpu().numpy().copy()

    x = diffusion.sample(model, batch, "ddpm", generator=generator, callback=record)
    x = x[0].cpu().numpy()
    raw = attach_doors(graph.id, decode_rooms(x, graph), graph)
    clean, _ = clean_plan(raw, graph)
    valid = validity(clean) == "valid"
    fig = Figure(figsize=(17, 2.9), dpi=120)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 8)
    draw_graph(graph, axes[0], seed)
    axes[0].set_title("1. requested room graph", fontsize=8)
    steps = [(999, "2. noise"), (600, "3. t = 600"), (300, "4. t = 300"), (100, "5. t = 100")]
    for ax, (t, label) in zip(axes[1:5], steps, strict=True):
        draw_corners(states[t], graph, ax, outline=t < 999)
        ax.set_title(f"{label}", fontsize=8)
    draw_plan(raw, axes[5], linewidth=0.8)
    axes[5].set_title("6. model output (t = 0)", fontsize=8)
    draw_plan(clean, axes[6], linewidth=0.8)
    axes[6].set_title("7. after clean-up", fontsize=8)
    if valid:
        axes[7].imshow(render_house(build_house(clean, building, roof="hip", seed=seed), "iso",
                                    (480, 360)))  # fmt: skip
    axes[7].axis("off")
    axes[7].set_title("8. 3D house", fontsize=8)
    fig.tight_layout(pad=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return valid


def hero_figure(results, building, path, count: int = 6) -> int:
    """3D view, plan and room graph of the first ``count`` valid houses of ``results``."""
    chosen = [r for r in results if r.house is not None][:count]
    if not chosen:
        return 0
    fig = Figure(figsize=(12, 2.6 * ((len(chosen) + 1) // 2)), dpi=120)
    FigureCanvasAgg(fig)
    rows = (len(chosen) + 1) // 2
    axes = np.atleast_2d(fig.subplots(rows, 6, gridspec_kw={"width_ratios": [2, 1.2, 1] * 2}))
    for ax in axes.ravel():
        ax.axis("off")
    for k, result in enumerate(chosen):
        row, col = divmod(k, 2)
        a_3d, a_plan, a_graph = axes[row, col * 3 : col * 3 + 3]
        a_3d.imshow(render_house(result.house, "iso", (640, 480)))
        draw_plan(result.clean, a_plan, linewidth=0.8)
        draw_graph(result.graph, a_graph)
    fig.tight_layout(pad=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return len(chosen)


def main() -> None:
    import yaml

    from housegen.gallery import generate_houses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/base/ema.pt")
    parser.add_argument("--data-config", default="configs/data/swiss_dwellings.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/figures")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, cfg = load_checkpoint(args.checkpoint, device)
    building = load_building_config()
    min_area = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))["min_area_m2"]
    test = load_split(load_config(args.data_config), "test")
    drawn = [RoomGraph.from_plan(p) for p in random.Random(args.seed).sample(test, 40)]
    out = Path(args.out)
    for k, graph in enumerate(drawn):  # the first graph of the draw whose plan is valid
        if pipeline_figure(model, diffusion, cfg, graph, building, out / "pipeline.png",
                           args.seed, device):  # fmt: skip
            print(f"pipeline figure: graph {graph.id} ({k + 1}th of the draw)")
            break
    results = generate_houses(model, diffusion, drawn, building, cfg, min_area, "ddpm",
                              None, args.seed, device)  # fmt: skip
    shown = hero_figure(results, building, out / "hero.png")
    first = [k for k, r in enumerate(results) if r.house is not None][:shown]
    print(f"hero figure: houses {first} of a random draw of {len(drawn)} test graphs")


if __name__ == "__main__":
    main()
