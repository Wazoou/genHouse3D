"""Quick look at a checkpoint during training: a few hundred validation graphs, DDIM.

Usage: ``python -m housegen.quick_check --checkpoint runs/base/ema_0020000.pt``

Not a substitute for ``housegen.experiments`` (test split, DDPM, five plans per graph): it
uses the validation split so that the test split is not looked at while choosing settings.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from housegen import metrics
from housegen.data.build import load_config, load_split
from housegen.graphs import RoomGraph
from housegen.postprocess import clean_plan, validity
from housegen.sample import generate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, cfg = load_checkpoint(args.checkpoint, device)
    data_cfg = load_config("configs/data/swiss_dwellings.yaml")
    val = random.Random(args.seed).sample(load_split(data_cfg, "val"), args.n)
    train = load_split(data_cfg, "train")
    graphs = [RoomGraph.from_plan(p) for p in val]
    min_area = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))["min_area_m2"]
    samples = generate(model, diffusion, graphs, 1, "ddim", args.steps, args.seed, device,
                       random_room_ids=cfg["augment"]["room_ids"])  # fmt: skip
    raw = [s[0] for s in samples]
    clean = [clean_plan(p, g)[0] for p, g in zip(raw, graphs, strict=True)]
    index = metrics.LayoutIndex(train)
    for name, plans in (("raw", raw), ("clean", clean)):
        verdicts = Counter(validity(p, min_area) for p in plans)
        distance = [metrics.compatibility(g, p)[0] for g, p in zip(graphs, plans, strict=True)]
        nearest = [index.nearest(metrics.label_map(p))[0] for p in plans]
        print(
            f"{name:5s} valid {verdicts['valid'] / len(plans):.1%}  "
            f"graph respected {np.mean([d == 0 for d in distance]):.1%}  "
            f"mean edit distance {np.mean(distance):.2f}  "
            f"nearest training plan {np.mean(nearest):.3f}  "
            f"copies {np.mean([s >= 0.9 for s in nearest]):.1%}  {dict(verdicts.most_common(4))}"
        )


if __name__ == "__main__":
    main()
