"""Dataset-independent filters: size limits, validity and duplicate removal."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from housegen.data.schema import Plan, validate_plan


@dataclass
class FilterConfig:
    min_rooms: int = 4
    max_rooms: int = 10
    max_corners_per_room: int = 12


def rejection_reason(plan: Plan, cfg: FilterConfig) -> str | None:
    """Why ``plan`` is filtered out, or ``None`` when it is kept."""
    if len(plan.rooms) < cfg.min_rooms:
        return "too_few_rooms"
    if len(plan.rooms) > cfg.max_rooms:
        return "too_many_rooms"
    if max(len(r.polygon) for r in plan.rooms) > cfg.max_corners_per_room:
        return "too_many_corners"
    problems = validate_plan(plan)
    if "no entrance door" in problems:
        return "no_entrance"
    if "room graph is not connected" in problems:
        return "disconnected_graph"
    if problems:
        return "invalid_plan"
    return None


def canonical_key(plan: Plan) -> tuple:
    """Key shared by plans that are equal up to translation, 90-degree rotation and mirroring.

    Room types and corner sets only: doors are ignored. Two apartments stacked on different
    floors, or mirrored on both sides of a staircase, get the same key.
    """
    keys = []
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                rooms = []
                for room in plan.rooms:
                    pts = [(y, x) if swap else (x, y) for x, y in room.polygon]
                    rooms.append((room.type, [(sx * x, sy * y) for x, y in pts]))
                x0 = min(x for _, pts in rooms for x, _ in pts)
                y0 = min(y for _, pts in rooms for _, y in pts)
                keys.append(
                    tuple(
                        sorted(
                            (t, tuple(sorted((x - x0, y - y0) for x, y in pts))) for t, pts in rooms
                        )
                    )
                )
    return min(keys)


def deduplicate(plans: Iterable[Plan]) -> tuple[list[Plan], int]:
    """Keep the first plan of every :func:`canonical_key`; return (kept, number removed)."""
    seen: set[tuple] = set()
    kept: list[Plan] = []
    removed = 0
    for plan in plans:
        key = canonical_key(plan)
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            kept.append(plan)
    return kept, removed
