"""Regenerate ``plans.jsonl`` from a built dataset: one training plan per room count.

Usage: ``python tests/fixtures/make_fixtures.py`` (after ``make data``).
"""

from __future__ import annotations

import random
from pathlib import Path

from housegen.data.build import load_config, load_split
from housegen.data.schema import write_plans


def main() -> None:
    train = load_split(load_config("configs/data/swiss_dwellings.yaml"), "train")
    picked: dict[int, object] = {}
    for plan in random.Random(0).sample(train, len(train)):
        picked.setdefault(len(plan.rooms), plan)
    plans = [picked[n] for n in sorted(picked)]
    write_plans(Path(__file__).parent / "plans.jsonl", plans)
    print([(p.id, len(p.rooms)) for p in plans])


if __name__ == "__main__":
    main()
