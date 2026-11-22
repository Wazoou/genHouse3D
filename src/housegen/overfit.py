"""Overfitting test (brief, section 4.3): the model must memorise 32 training plans.

Usage: ``python -m housegen.overfit --config configs/model/overfit.yaml``

Trains on the 32 plans without augmentation, then regenerates a plan for each of their
graphs with DDPM (1000 steps) and DDIM (100 steps) and measures how closely the targets are
reproduced. Writes ``reports/results/overfit.csv`` and figures in ``reports/figures/model/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen import metrics
from housegen.graphs import RoomGraph
from housegen.model.encoding import plan_corners, to_grid
from housegen.postprocess import attach_doors
from housegen.sample import generate, load_checkpoint
from housegen.train import load_train_config, run, select_plans
from housegen.viz import plot_plans


def corner_errors(target, generated) -> np.ndarray:
    """Distance in grid units between matching corners (same room, same corner index)."""
    a = to_grid(plan_corners(target)[0])
    b = np.concatenate([np.asarray(r.polygon, dtype=float) for r in generated.rooms])
    return np.hypot(*(a - b).T)


def loss_figure(log_path: Path, path: Path) -> None:
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig = Figure(figsize=(6.4, 3.4), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    train = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    ax.plot(*zip(*train, strict=True), label="training loss (batch mean)", linewidth=1.2)
    if val:
        ax.plot(*zip(*val, strict=True), "o-", label="EMA model, fixed noise", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("noise prediction MSE")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/overfit.yaml")
    parser.add_argument("--out", default="runs/overfit")
    parser.add_argument("--device", default=None)
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--figures", default="reports/figures/model")
    parser.add_argument("--skip-training", action="store_true", help="reuse <out>/ema.pt")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_train_config(args.config)
    plans, _ = select_plans(cfg)
    out = Path(args.out)
    if not args.skip_training:
        print(run(cfg, out, device))
    model, diffusion, cfg = load_checkpoint(out / "ema.pt", device)
    graphs = [RoomGraph.from_plan(p) for p in plans]
    with open(out / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    final_loss = float(rows[-1]["loss"])
    final_val = next(float(r["val_loss"]) for r in reversed(rows) if r["val_loss"])
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    summary = []
    for method, steps in (("ddpm", None), ("ddim", 100)):
        samples = generate(
            model, diffusion, graphs, 1, method, steps, seed=0, device=device,
            random_room_ids=cfg["augment"]["room_ids"],
        )  # fmt: skip
        generated = [s[0] for s in samples]
        errors = np.concatenate(
            [corner_errors(t, g) for t, g in zip(plans, generated, strict=True)]
        )
        similarity = [
            metrics.layout_similarity(metrics.label_map(t), metrics.label_map(g))
            for t, g in zip(plans, generated, strict=True)
        ]
        distance = [
            metrics.compatibility(gr, g)[0] for gr, g in zip(graphs, generated, strict=True)
        ]
        # Raw corners are a few centimetres off, so neighbouring walls rarely coincide and
        # doors cannot be placed; with a 20 cm tolerance on shared walls they can.
        tolerant = [
            metrics.compatibility(gr, attach_doors(g.id, g.rooms, gr, tol=2.0))[0]
            for gr, g in zip(graphs, generated, strict=True)
        ]
        summary.append(
            {
                "sampler": f"{method} ({diffusion.steps if method == 'ddpm' else steps} steps)",
                "plans": len(plans),
                "final_training_loss": round(final_loss, 5),
                "final_ema_loss_fixed_noise": round(final_val, 5),
                "corners_exact": round(float((errors < 0.5).mean()), 4),
                "corners_within_1_unit": round(float((errors <= 1.0 + 1e-9).mean()), 4),
                "mean_corner_error_m": round(float(errors.mean()) * plans[0].meters_per_unit, 4),
                "max_corner_error_m": round(float(errors.max()) * plans[0].meters_per_unit, 3),
                "mean_layout_similarity": round(float(np.mean(similarity)), 4),
                "min_layout_similarity": round(float(np.min(similarity)), 4),
                "graph_compatible_rate": round(float(np.mean([d == 0 for d in distance])), 4),
                "graph_compatible_rate_20cm": round(float(np.mean([d == 0 for d in tolerant])), 4),
            }
        )
        print(summary[-1])
        if method == "ddpm":
            pairs = [p for t, g in zip(plans[:8], generated[:8], strict=True) for p in (t, g)]
            titles = [x for t in plans[:8] for x in ("training plan", "regenerated (DDPM)")]
            plot_plans(pairs, figures / "overfit_samples.png", ncols=4, titles=titles)
    path = Path(args.results) / "overfit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    loss_figure(out / "log.csv", figures / "overfit_loss.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
