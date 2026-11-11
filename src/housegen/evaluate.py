"""``make evaluate``: score methods on the test graphs and write ``reports/results/*.csv``.

Usage: ``python -m housegen.evaluate --config configs/data/swiss_dwellings.yaml``

Two methods are always scored: the real test plans (a reference and a sanity check: their
doors are stripped and re-derived from the geometry, and their compatibility must be
perfect) and the retrieval baseline B0. Generated samples are added with
``--samples NAME=plans.jsonl.gz`` where each plan carries ``meta.condition`` (the id of the
test plan whose graph it was generated for).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen import metrics
from housegen.baselines.retrieval import RetrievalBaseline
from housegen.data.build import load_config, load_split
from housegen.data.schema import ROOM_TYPES, Plan, iter_plans
from housegen.graphs import RoomGraph
from housegen.postprocess import attach_doors, validity

Features = Callable[[np.ndarray], np.ndarray]


def load_eval_config(path: str | Path = "configs/eval.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def rederive_doors(plan: Plan, cfg: dict) -> Plan:
    """``plan`` with its doors and openings rebuilt from its rooms and its own graph."""
    d = cfg["doors"]
    return attach_doors(
        plan.id, plan.rooms, RoomGraph.from_plan(plan), d["width"], d["entrance_width"],
        d["min_wall"], source=plan.source, group=plan.group,
    )  # fmt: skip


def evaluate_method(
    name: str,
    conditions: Sequence[RoomGraph],
    samples: Sequence[Sequence[Plan]],
    real: Sequence[Plan],
    index: metrics.LayoutIndex,
    cfg: dict,
    real_features: np.ndarray | None = None,
    features: Features | None = None,
) -> tuple[dict, list[dict]]:
    """All metrics of one method: ``samples[k]`` are the plans produced for ``conditions[k]``.

    Returns the summary row and one row per generated plan.
    """
    rows, diversities = [], []
    for graph, plans in zip(conditions, samples, strict=True):
        maps = [metrics.label_map(p) for p in plans]
        if len(maps) > 1:
            diversities.append(metrics.diversity(maps))
        for k, (plan, labels) in enumerate(zip(plans, maps, strict=True)):
            distance, proven = metrics.compatibility(graph, plan)
            similarity, nearest = index.nearest(labels)
            rows.append(
                {
                    "method": name,
                    "condition": graph.id,
                    "sample": k,
                    "validity": validity(plan, cfg["min_area_m2"]),
                    "graph_edit_distance": distance,
                    "distance_proven": int(proven),
                    "nearest_train_similarity": round(similarity, 4),
                    "nearest_train_plan": nearest,
                }
            )
    generated = [p for plans in samples for p in plans]
    distances = np.array([r["graph_edit_distance"] for r in rows])
    similarities = np.array([r["nearest_train_similarity"] for r in rows])
    w1 = metrics.area_distance(real, generated)
    fid = float("nan")
    if features is not None and real_features is not None:
        fake = features(metrics.render_for_fid(generated, cfg["fid"]["image_size"]))
        fid = metrics.frechet_distance(real_features, fake)
    summary = {
        "method": name,
        "graphs": len(conditions),
        "plans": len(generated),
        "fid": round(fid, 2) + 0.0,  # avoids "-0.0" for identical sets
        "compatibility_ged": round(float(distances.mean()), 3),
        "compatible_rate": round(float((distances == 0).mean()), 4),
        "ged_not_proven": int(sum(1 - r["distance_proven"] for r in rows)),
        "valid_rate": round(float(np.mean([r["validity"] == "valid" for r in rows])), 4),
        "diversity": round(float(np.mean(diversities)), 4) if diversities else float("nan"),
        "nearest_train_similarity": round(float(similarities.mean()), 4),
        "copy_rate": round(float((similarities >= cfg["copy_threshold"]).mean()), 4),
        "area_w1_m2": round(float(np.nanmean(list(w1.values()))), 3),
    }
    return summary, rows


def load_samples(path: str | Path, conditions: Sequence[RoomGraph]) -> list[list[Plan]]:
    """Generated plans grouped by ``meta.condition``, in the order of ``conditions``."""
    groups: dict[str, list[Plan]] = defaultdict(list)
    for plan in iter_plans(path):
        groups[plan.meta["condition"]].append(plan)
    return [groups[g.id] for g in conditions]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def novelty_figure(rows: list[dict], path: Path, threshold: float) -> None:
    """Histogram of the similarity to the nearest training plan, one curve per method."""
    fig = Figure(figsize=(6.4, 3.6), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    bins = np.linspace(0.0, 1.0, 41)
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row["nearest_train_similarity"])
    for method, values in by_method.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=1.6, density=True, label=method)
    ax.axvline(threshold, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("similarity to the nearest training plan (1 = copy)")
    ax.set_ylabel("density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def run(
    data_cfg: dict,
    cfg: dict,
    out: Path,
    figures: Path,
    extra_samples: dict[str, str] | None = None,
    features: Features | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Evaluate the reference, B0 and any extra sample files; write the result tables."""
    train, test = load_split(data_cfg, "train"), load_split(data_cfg, "test")
    if limit:
        test = test[:limit]
    conditions = [RoomGraph.from_plan(p) for p in test]
    index = metrics.LayoutIndex(train)
    real_features = None
    if features is not None:
        real_features = features(metrics.render_for_fid(test, cfg["fid"]["image_size"]))
    baseline = RetrievalBaseline(train, cfg["retrieval"]["shortlist"])
    methods: dict[str, list[list[Plan]]] = {
        "real_test": [[rederive_doors(p, cfg)] for p in test],
        "b0_retrieval": [baseline.sample(g, cfg["samples_per_graph"]) for g in conditions],
    }
    for name, path in (extra_samples or {}).items():
        methods[name] = load_samples(path, conditions)
    summaries, all_rows = [], []
    for name, samples in methods.items():
        summary, rows = evaluate_method(
            name, conditions, samples, test, index, cfg, real_features, features
        )
        summaries.append(summary)
        all_rows += rows
        print(summary)
    _write_csv(out / "metrics.csv", summaries)
    _write_csv(out / "metrics_per_plan.csv", all_rows)
    reasons = Counter((r["method"], r["validity"]) for r in all_rows)
    totals = Counter(r["method"] for r in all_rows)
    _write_csv(
        out / "validity_reasons.csv",
        [
            {"method": m, "verdict": v, "plans": n, "share": round(n / totals[m], 4)}
            for (m, v), n in sorted(reasons.items(), key=lambda kv: (kv[0][0], -kv[1]))
        ],
    )
    real_areas = metrics.room_areas(test)
    area_rows = []
    for name, samples in methods.items():
        generated = [p for plans in samples for p in plans]
        areas, w1 = metrics.room_areas(generated), metrics.area_distance(test, generated)
        for t in ROOM_TYPES:
            area_rows.append(
                {
                    "method": name,
                    "room_type": t,
                    "rooms": len(areas[t]),
                    "median_m2": round(float(np.median(areas[t])), 2) if len(areas[t]) else "",
                    "real_median_m2": round(float(np.median(real_areas[t])), 2)
                    if len(real_areas[t])
                    else "",
                    "wasserstein_m2": round(w1[t], 3),
                }
            )
    _write_csv(out / "areas_by_type.csv", area_rows)
    novelty_figure(all_rows, figures / "novelty_hist.png", cfg["copy_threshold"])
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/data/<dataset>.yaml")
    parser.add_argument("--eval", default="configs/eval.yaml")
    parser.add_argument("--out", default="reports/results")
    parser.add_argument("--figures", default="reports/figures/eval")
    parser.add_argument("--samples", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--no-fid", action="store_true", help="skip the FID (no PyTorch needed)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N test graphs")
    args = parser.parse_args()
    features = None if args.no_fid else metrics.InceptionFeatures()
    run(
        load_config(args.config),
        load_eval_config(args.eval),
        Path(args.out),
        Path(args.figures),
        dict(s.split("=", 1) for s in args.samples),
        features,
        args.limit,
    )


if __name__ == "__main__":
    main()
