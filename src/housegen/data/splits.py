"""Leakage-free train / val / test split: all plans of a group share the same split."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from housegen.data.schema import Plan

SPLITS: tuple[str, ...] = ("train", "val", "test")


def split_by_group(
    plans: Sequence[Plan],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[str]]:
    """Assign whole groups (buildings / sites) to splits; return plan ids per split.

    Groups are visited in a seeded random order and each one goes to the split that is the
    furthest below its target share of plans, so the plan counts follow ``fractions`` even
    though groups have very different sizes.
    """
    sizes = Counter(p.group for p in plans)
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    total = sum(sizes.values())
    counts = [0] * len(SPLITS)
    assignment: dict[str, int] = {}
    for group in groups:
        deficit = [fractions[k] - counts[k] / total for k in range(len(SPLITS))]
        k = max(range(len(SPLITS)), key=lambda k: deficit[k] / fractions[k])
        assignment[group] = k
        counts[k] += sizes[group]
    out: dict[str, list[str]] = {name: [] for name in SPLITS}
    for plan in plans:
        out[SPLITS[assignment[plan.group]]].append(plan.id)
    return {name: sorted(ids) for name, ids in out.items()}


def split_path(splits_dir: str | Path, dataset: str, split: str) -> Path:
    return Path(splits_dir) / f"{dataset}_{split}.txt"


def write_splits(splits_dir: str | Path, dataset: str, splits: dict[str, list[str]]) -> None:
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        split_path(splits_dir, dataset, name).write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_split(splits_dir: str | Path, dataset: str, split: str) -> list[str]:
    text = split_path(splits_dir, dataset, split).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line]
