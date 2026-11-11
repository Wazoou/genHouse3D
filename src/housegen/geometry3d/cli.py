"""``make build3d``: real plans -> GLB houses, mesh validation and roof statistics.

Usage: ``python -m housegen.geometry3d.cli --config configs/data/swiss_dwellings.yaml``

Writes one GLB with roof and one cutaway GLB per sampled plan under ``outputs/``, the
per-house report ``reports/results/build3d_real.csv`` and the roof success rates over the
whole split in ``reports/results/roof_stats_real.csv``.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from housegen.data.build import load_config, load_split
from housegen.data.schema import Plan
from housegen.geometry3d.export import build_house, load_building_config

GALLERY_ROOFS = ("hip", "hip", "gable", "flat")


def sample_plans(data_cfg: dict, split: str, n: int, seed: int) -> list[Plan]:
    """``n`` plans drawn at random (seeded) from a split: no hand picking."""
    plans = load_split(data_cfg, split)
    return random.Random(seed).sample(plans, min(n, len(plans)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_houses(plans: list[Plan], cfg: dict, out_dir: Path, seed: int) -> list[dict]:
    """Build, validate and export each plan with a roof and as a cutaway."""
    rows = []
    for k, plan in enumerate(plans):
        roof = GALLERY_ROOFS[k % len(GALLERY_ROOFS)]
        start = time.perf_counter()
        house = build_house(plan, cfg, roof=roof, seed=seed)
        cutaway = build_house(plan, cfg, roof="none", seed=seed)
        seconds = time.perf_counter() - start
        problems = house.problems() + cutaway.problems()
        glb = house.export(out_dir / f"{plan.id}.glb")
        cutaway.export(out_dir / f"{plan.id}_cutaway.glb")
        rows.append(
            {
                "plan_id": plan.id,
                "rooms": len(plan.rooms),
                "roof_requested": roof,
                "roof_built": house.info["roof_built"],
                "roof_note": house.info["roof_error"],
                "valid": int(not problems),
                "problems": "; ".join(problems),
                "doors": house.info["doors"],
                "windows": house.info["windows"],
                "triangles": house.info["triangles"],
                "glb_kb": round(glb.stat().st_size / 1024, 1),
                "build_seconds": round(seconds, 3),
            }
        )
    return rows


def roof_statistics(plans: list[Plan], cfg: dict, seed: int) -> list[dict]:
    """For each roof type: how often it is built as requested, and how often meshes are valid."""
    rows = []
    for roof in ("hip", "gable", "flat"):
        built = valid = 0
        start = time.perf_counter()
        for plan in plans:
            house = build_house(plan, cfg, roof=roof, seed=seed)
            built += house.info["roof_built"] == roof
            valid += not house.problems()
        rows.append(
            {
                "roof_requested": roof,
                "plans": len(plans),
                "built_as_requested": built,
                "fallbacks": len(plans) - built,
                "failure_rate": round(1 - built / len(plans), 4),
                "valid_meshes": valid,
                "mean_build_seconds": round((time.perf_counter() - start) / len(plans), 3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/data/<dataset>.yaml")
    parser.add_argument("--building", default="configs/building.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/houses_real")
    parser.add_argument("--results", default="reports/results")
    args = parser.parse_args()
    cfg, data_cfg = load_building_config(args.building), load_config(args.config)
    rows = export_houses(sample_plans(data_cfg, args.split, args.n, args.seed), cfg,
                         Path(args.out), args.seed)  # fmt: skip
    _write_csv(Path(args.results) / "build3d_real.csv", rows)
    print(f"{sum(r['valid'] for r in rows)} / {len(rows)} valid houses exported to {args.out}")
    stats = roof_statistics(load_split(data_cfg, args.split), cfg, args.seed)
    _write_csv(Path(args.results) / "roof_stats_real.csv", stats)
    for row in stats:
        print(row)


if __name__ == "__main__":
    main()
