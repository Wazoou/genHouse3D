"""``make data``: raw dataset -> unified plans, filters, de-duplication and frozen split.

Usage: ``python -m housegen.data.build --config configs/data/swiss_dwellings.yaml``
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import yaml

from housegen.data import swiss_dwellings
from housegen.data.filters import FilterConfig, deduplicate, rejection_reason
from housegen.data.schema import Plan, iter_plans, write_plans
from housegen.data.splits import SPLITS, read_split, split_by_group, write_splits


def _swiss_dwellings(cfg: dict, rejected: Counter) -> Iterator[Plan]:
    adapter_cfg = swiss_dwellings.SwissDwellingsConfig(
        type_map=cfg["type_map"],
        ignored_subtypes=frozenset(cfg["ignored_subtypes"]),
        **cfg.get("geometry", {}),
    )
    return swiss_dwellings.iter_plans(cfg["raw_csv"], adapter_cfg, rejected)


# One adapter per dataset: a callable (config, rejection counter) -> iterator of plans.
ADAPTERS = {"swiss_dwellings": _swiss_dwellings}


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def plans_path(cfg: dict) -> Path:
    return Path(cfg["processed_dir"]) / "plans.jsonl.gz"


def load_split(cfg: dict, split: str) -> list[Plan]:
    """Plans of one frozen split (``train``, ``val`` or ``test``) of a built dataset."""
    ids = set(read_split(cfg["splits_dir"], cfg["name"], split))
    return [p for p in iter_plans(plans_path(cfg)) if p.id in ids]


def build(cfg: dict) -> dict:
    """Run the full preprocessing; write plans, split files and statistics; return the stats."""
    start = time.time()
    rejected: Counter = Counter()
    filter_cfg = FilterConfig(**cfg.get("filters", {}))
    converted = 0
    kept: list[Plan] = []
    seen_ids: set[str] = set()
    for plan in ADAPTERS[cfg["adapter"]](cfg, rejected):
        converted += 1
        reason = rejection_reason(plan, filter_cfg)
        if reason is not None:
            rejected[reason] += 1
            continue
        if plan.id in seen_ids:
            raise ValueError(f"duplicate plan id {plan.id}")
        seen_ids.add(plan.id)
        kept.append(plan)
    unique, duplicates = deduplicate(kept)
    split_cfg = cfg.get("split", {})
    splits = split_by_group(
        unique, split_cfg.get("fractions", (0.8, 0.1, 0.1)), split_cfg.get("seed", 0)
    )
    write_plans(plans_path(cfg), unique)
    write_splits(cfg["splits_dir"], cfg["name"], splits)
    groups = {p.id: p.group for p in unique}
    stats = {
        "dataset": cfg["name"],
        "raw_dwellings": converted
        + sum(v for k, v in rejected.items() if k not in _FILTER_REASONS),
        "converted": converted,
        "rejected": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
        "duplicates_removed": duplicates,
        "kept": len(unique),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "groups": {name: len({groups[i] for i in ids}) for name, ids in splits.items()},
        "seconds": round(time.time() - start, 1),
    }
    stats_path = Path(cfg["processed_dir"]) / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


_FILTER_REASONS = {
    "too_few_rooms",
    "too_many_rooms",
    "too_many_corners",
    "no_entrance",
    "disconnected_graph",
    "invalid_plan",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/data/<dataset>.yaml")
    args = parser.parse_args()
    stats = build(load_config(args.config))
    print(json.dumps(stats, indent=2))
    assert set(stats["splits"]) == set(SPLITS)


if __name__ == "__main__":
    main()
