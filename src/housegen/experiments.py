"""Sample, clean and evaluate trained models on the test graphs (``make experiments``).

Usage::

    python -m housegen.experiments --model base=runs/base/ema.pt [--model b1=...] \
        [--variant no_rounding=base]

For each model: 5 plans per test graph with DDPM, written raw (``<name>_raw``) and after
clean-up (``<name>_clean``) under ``outputs/samples/``; a ``--variant no_rounding=NAME``
also samples NAME without the final rounding to the grid. All sample files are then scored
together with the references by ``housegen.evaluate`` (tables in ``reports/results/``).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from housegen import metrics
from housegen.data.build import load_config, load_split
from housegen.data.schema import iter_plans, write_plans
from housegen.evaluate import load_eval_config, run
from housegen.graphs import RoomGraph
from housegen.postprocess import clean_plan, requested_graph
from housegen.sample import generate, load_checkpoint


def sample_split(
    checkpoint: str | Path, graphs: list[RoomGraph], out: Path, per_graph: int, method: str,
    steps: int | None, seed: int, device: str, discretize: bool = True,
) -> dict:  # fmt: skip
    """Generate and write plans for ``graphs``; return timing information."""
    model, diffusion, cfg = load_checkpoint(checkpoint, device)
    start = time.perf_counter()
    samples = generate(
        model, diffusion, graphs, per_graph, method, steps, seed, device, batch_size=1024,
        random_room_ids=cfg["augment"]["room_ids"], discretize=discretize,
    )  # fmt: skip
    seconds = time.perf_counter() - start
    write_plans(out, [p for group in samples for p in group])
    return {"seconds": round(seconds, 1), "plans": len(graphs) * per_graph, "device": device}


def clean_file(raw: Path, out: Path) -> dict:
    """Clean every plan of ``raw`` into ``out``; return the summed clean-up counters."""
    totals: Counter = Counter()

    def cleaned():
        for plan in iter_plans(raw):
            result, stats = clean_plan(plan, requested_graph(plan))
            totals.update(stats)
            yield result

    write_plans(out, cleaned())
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--variant", action="append", default=[], metavar="no_rounding=NAME")
    parser.add_argument("--data-config", default="configs/data/swiss_dwellings.yaml")
    parser.add_argument("--eval", default="configs/eval.yaml")
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", default="outputs/samples")
    parser.add_argument("--out", default="reports/results")
    parser.add_argument("--figures", default="reports/figures/eval")
    parser.add_argument("--resample", action="store_true", help="ignore existing sample files")
    parser.add_argument("--no-fid", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cfg = load_eval_config(args.eval)
    test = load_split(load_config(args.data_config), "test")
    graphs = [RoomGraph.from_plan(p) for p in test]
    per_graph = eval_cfg["samples_per_graph"]
    steps = None if args.method == "ddpm" else args.steps
    folder = Path(args.samples)
    models = dict(m.split("=", 1) for m in args.model)
    jobs = [(name, ckpt, True) for name, ckpt in models.items()]
    for item in args.variant:
        kind, name = item.split("=", 1)
        if kind != "no_rounding":
            raise ValueError(f"unknown variant {kind}")
        jobs.append((f"{name}_no_rounding", models[name], False))
    files, log = {}, {}
    for name, ckpt, discretize in jobs:
        raw = folder / f"{name}_{args.method}_raw.jsonl.gz"
        clean = folder / f"{name}_{args.method}_clean.jsonl.gz"
        if args.resample or not raw.exists():
            log[name] = sample_split(
                ckpt, graphs, raw, per_graph, args.method, steps, args.seed, device, discretize
            )
            print(name, log[name])
        files[f"{name}_raw"] = str(raw)
        if discretize:
            log[f"{name}_cleanup"] = clean_file(raw, clean)
            files[f"{name}_clean"] = str(clean)
    features = None if args.no_fid else metrics.InceptionFeatures()
    run(
        load_config(args.data_config), eval_cfg, Path(args.out), Path(args.figures), files, features
    )
    path = Path(args.out) / "experiments_log.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps(previous | log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
