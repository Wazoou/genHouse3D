"""Generated houses end to end: graph -> diffusion -> clean-up -> 3D, and the honest gallery.

Usage (``make render``):
``python -m housegen.gallery --checkpoint runs/base/ema.pt --n 24``

Draws ``n`` test graphs at random (seeded, not hand-picked), generates one plan for each,
cleans it, extrudes the valid ones in 3D and exports their GLB files. Every drawn graph
appears in the gallery: invalid plans are shown as plans, with the reason. A second figure
shows typical failures from a larger draw.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen.data.build import load_config, load_split
from housegen.data.schema import Plan
from housegen.geometry3d.export import House, build_house, load_building_config
from housegen.graphs import RoomGraph
from housegen.metrics import compatibility
from housegen.postprocess import clean_plan, validity
from housegen.render import render_house
from housegen.sample import generate, load_checkpoint
from housegen.viz import draw_plan

ROOFS = ("hip", "hip", "gable", "flat")


@dataclass
class Result:
    """One generated house and what happened to it along the chain."""

    graph: RoomGraph
    raw: Plan
    clean: Plan
    verdict: str
    raw_verdict: str
    graph_edit_distance: int
    house: House | None = None
    error: str = ""
    info: dict = field(default_factory=dict)


def generate_houses(
    model, diffusion, graphs: list[RoomGraph], building: dict, cfg: dict, min_area: dict,
    method: str = "ddpm", steps: int | None = None, seed: int = 0, device: str = "cpu",
) -> list[Result]:  # fmt: skip
    """One house per graph; the roof type cycles through ``ROOFS``."""
    samples = generate(
        model, diffusion, graphs, 1, method, steps, seed, device,
        random_room_ids=cfg["augment"]["room_ids"],
    )  # fmt: skip
    results = []
    for k, (graph, [raw]) in enumerate(zip(graphs, samples, strict=True)):
        clean, _ = clean_plan(raw, graph)
        result = Result(
            graph, raw, clean, validity(clean, min_area), validity(raw, min_area),
            compatibility(graph, clean)[0],
        )  # fmt: skip
        if result.verdict == "valid":
            try:
                result.house = build_house(clean, building, roof=ROOFS[k % len(ROOFS)], seed=seed)
                result.info = result.house.info
            except Exception as error:  # noqa: BLE001 - reported, never hidden
                result.error = f"{type(error).__name__}: {error}"
        results.append(result)
    return results


def _caption(result: Result) -> str:
    rooms = len(result.graph.types)
    if result.house is None:
        return f"{rooms} rooms - not built: {result.error or result.verdict}"
    compatible = "graph respected" if result.graph_edit_distance == 0 else (
        f"graph distance {result.graph_edit_distance}"
    )  # fmt: skip
    return f"{rooms} rooms, {result.info['roof_built']} roof, {compatible}"


def gallery_figure(results: list[Result], building: dict, path: Path, per_row: int = 2) -> None:
    """For each house: 3/4 view with roof, view from above without roof, cleaned 2D plan."""
    rows = -(-len(results) // per_row)
    fig = Figure(figsize=(per_row * 3 * 2.6, rows * 2.3), dpi=110)
    FigureCanvasAgg(fig)
    axes = np.atleast_2d(fig.subplots(rows, per_row * 3)).reshape(rows, per_row * 3)
    for ax in axes.ravel():
        ax.axis("off")
    for k, result in enumerate(results):
        row, col = divmod(k, per_row)
        a_iso, a_top, a_plan = axes[row, col * 3 : col * 3 + 3]
        if result.house is not None:
            cutaway = build_house(result.clean, building, roof="none")
            a_iso.imshow(render_house(result.house, "iso", (560, 420)))
            a_top.imshow(render_house(cutaway, "top", (560, 420)))
        else:
            draw_plan(result.raw, a_iso, linewidth=0.8)
            a_iso.set_title("raw model output", fontsize=6)
            a_top.text(0.5, 0.5, f"not extruded\n({result.verdict})", ha="center",
                       va="center", fontsize=8, color="#a33")  # fmt: skip
        draw_plan(result.clean, a_plan, linewidth=0.8)
        a_plan.set_title(_caption(result), fontsize=6)
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def failures_figure(results: list[Result], path: Path, count: int = 6) -> list[Result]:
    """Raw output and clean-up of the first ``count`` failed plans, with the reason."""
    failed = [r for r in results if r.verdict != "valid" or r.error][:count]
    if not failed:
        return []
    fig = Figure(figsize=(2 * 2.8, len(failed) * 2.6), dpi=110)
    FigureCanvasAgg(fig)
    axes = np.atleast_2d(fig.subplots(len(failed), 2))
    for (a_raw, a_clean), result in zip(axes, failed, strict=True):
        draw_plan(result.raw, a_raw, linewidth=0.8)
        draw_plan(result.clean, a_clean, linewidth=0.8)
        a_raw.set_title(f"raw: {result.raw_verdict}", fontsize=7)
        a_clean.set_title(f"after clean-up: {result.verdict}", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/base/ema.pt")
    parser.add_argument("--data-config", default="configs/data/swiss_dwellings.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--building", default="configs/building.yaml")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--failure-pool", type=int, default=200)
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="generated")
    parser.add_argument("--figures", default="reports/figures/gallery")
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--glb", default="outputs/houses_generated")
    args = parser.parse_args()
    import yaml

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg = load_checkpoint(args.checkpoint, device)
    building = load_building_config(args.building)
    min_area = yaml.safe_load(Path(args.eval_config).read_text(encoding="utf-8"))["min_area_m2"]
    test = load_split(load_config(args.data_config), "test")
    drawn = random.Random(args.seed).sample(test, args.failure_pool)
    graphs = [RoomGraph.from_plan(p) for p in drawn]
    results = generate_houses(
        model, diffusion, graphs, building, cfg, min_area, args.method,
        None if args.method == "ddpm" else args.steps, args.seed, device,
    )  # fmt: skip
    shown = results[: args.n]  # the first n of a seeded random draw: no hand picking
    figures, glb = Path(args.figures), Path(args.glb)
    gallery_figure(shown, building, figures / f"gallery_{args.name}.png")
    gallery_figure(shown[:6], building, figures / f"gallery_{args.name}_small.png")
    failures_figure(results, figures / f"failures_{args.name}.png")
    rows = []
    for k, result in enumerate(results):
        if k < args.n and result.house is not None:
            result.house.export(glb / f"{result.graph.id}.glb")
            build_house(result.clean, building, roof="none").export(
                glb / f"{result.graph.id}_cutaway.glb"
            )
        rows.append(
            {
                "graph": result.graph.id,
                "in_gallery": int(k < args.n),
                "rooms": len(result.graph.types),
                "raw_verdict": result.raw_verdict,
                "verdict": result.verdict,
                "graph_edit_distance": result.graph_edit_distance,
                "built_3d": int(result.house is not None),
                "roof": result.info.get("roof_built", ""),
                "error": result.error,
            }
        )
    path = Path(args.results) / f"gallery_{args.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(Counter(r["verdict"] for r in rows), "built:", sum(r["built_3d"] for r in rows))


if __name__ == "__main__":
    main()
