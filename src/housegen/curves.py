"""Training curves from the CSV logs written by ``housegen.train``.

Usage: ``python -m housegen.curves base=runs/base/log.csv [NAME=PATH ...] --out curves.png``
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def read_log(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def training_curves(logs: dict[str, str | Path], path: str | Path, smooth: int = 10) -> None:
    """Training loss (moving average of ``smooth`` log rows) and validation loss per run."""
    fig = Figure(figsize=(7.0, 3.6), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    colours = ["#2f6db3", "#d9822b", "#3b9b5a", "#a2449b", "#7a7a7a"]
    for (name, log), colour in zip(logs.items(), colours, strict=False):
        rows = read_log(log)
        train = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
        steps = [s for s, _ in train]
        values = [v for _, v in train]
        smoothed = [
            sum(values[max(0, k - smooth + 1) : k + 1])
            / len(values[max(0, k - smooth + 1) : k + 1])
            for k in range(len(values))
        ]
        ax.plot(steps, smoothed, color=colour, linewidth=1.2, label=f"{name}: training")
        val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
        if val:
            ax.plot(*zip(*val, strict=True), "o--", color=colour, markersize=3, linewidth=1,
                    label=f"{name}: validation (EMA, fixed noise)")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("noise prediction MSE")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="NAME=PATH or PATH (name: parent folder)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logs = {}
    for item in args.logs:
        name, _, path = item.rpartition("=")
        logs[name or Path(path).parent.name] = path
    training_curves(logs, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
