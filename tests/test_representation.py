"""Unified format: serialisation, validity checks, graph, filters, duplicates, split."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import square_plan
from housegen.data.filters import FilterConfig, canonical_key, deduplicate, rejection_reason
from housegen.data.schema import (
    GRID_SIZE,
    ROOM_TYPES,
    Door,
    Plan,
    Room,
    read_plans,
    validate_plan,
    write_plans,
)
from housegen.data.splits import read_split, split_by_group, write_splits


def test_real_fixture_plans_are_valid(real_plans):
    assert len(real_plans) == 7
    assert [len(p.rooms) for p in real_plans] == [4, 5, 6, 7, 8, 9, 10]
    for plan in real_plans:
        assert validate_plan(plan) == []
        assert all(r.type in ROOM_TYPES for r in plan.rooms)
        for room in plan.rooms:
            assert all(isinstance(v, int) and 0 <= v < GRID_SIZE for p in room.polygon for v in p)
            assert len(room.polygon) % 2 == 0  # rectilinear rings


def test_round_trip(tmp_path, real_plans):
    for name in ("plans.jsonl", "plans.jsonl.gz"):
        assert write_plans(tmp_path / name, real_plans) == len(real_plans)
        assert read_plans(tmp_path / name) == real_plans


def test_graph_and_areas():
    plan = square_plan()
    assert plan.edges() == [(0, 1, "open"), (0, 2, "door"), (2, 3, "door")]
    g = plan.graph()
    assert [g.nodes[i]["type"] for i in g] == ["living", "kitchen", "bedroom", "bathroom"]
    assert g.edges[0, 2]["kind"] == "door" and g.edges[0, 1]["kind"] == "open"
    assert plan.entrance_rooms() == [0]
    assert plan.room_area_m2(0) == pytest.approx(16.0)  # 40 x 40 units of 10 cm


def test_a_door_wins_over_an_opening():
    plan = square_plan()
    plan = dataclasses.replace(plan, open_edges=((0, 1), (2, 0)))
    assert (0, 2, "door") in plan.edges()


def test_validate_detects_problems():
    plan = square_plan()
    assert validate_plan(plan) == []

    overlapping = list(plan.rooms)
    overlapping[1] = Room("kitchen", ((130, 100), (180, 100), (180, 140), (130, 140)))
    assert "rooms 0 and 1 overlap" in validate_plan(
        dataclasses.replace(plan, rooms=tuple(overlapping))
    )

    assert "room graph is not connected" in validate_plan(dataclasses.replace(plan, open_edges=()))

    no_entrance = tuple(d for d in plan.doors if not d.is_entrance)
    assert "no entrance door" in validate_plan(dataclasses.replace(plan, doors=no_entrance))

    off_wall = (*plan.doors, Door((1, 3), ((150, 120), (160, 120))))
    assert any(
        "not on a wall" in m for m in validate_plan(dataclasses.replace(plan, doors=off_wall))
    )

    outside = list(plan.rooms)
    outside[3] = Room("bathroom", ((140, 140), (300, 140), (300, 180), (140, 180)))
    assert any(
        "outside the grid" in m
        for m in validate_plan(dataclasses.replace(plan, rooms=tuple(outside)))
    )

    clockwise = list(plan.rooms)
    clockwise[0] = Room("living", plan.rooms[0].polygon[::-1])
    assert any(
        "counter-clockwise" in m
        for m in validate_plan(dataclasses.replace(plan, rooms=tuple(clockwise)))
    )


def test_filters(real_plans):
    cfg = FilterConfig(min_rooms=4, max_rooms=10, max_corners_per_room=12)
    assert all(rejection_reason(p, cfg) is None for p in real_plans)
    assert rejection_reason(real_plans[0], FilterConfig(min_rooms=5)) == "too_few_rooms"
    assert rejection_reason(real_plans[-1], FilterConfig(max_rooms=9)) == "too_many_rooms"
    assert (
        rejection_reason(real_plans[-1], FilterConfig(max_corners_per_room=4)) == "too_many_corners"
    )
    plan = square_plan()
    assert rejection_reason(dataclasses.replace(plan, open_edges=()), cfg) == "disconnected_graph"
    assert rejection_reason(dataclasses.replace(plan, doors=plan.doors[:2]), cfg) == "no_entrance"


def _transform(plan: Plan, fn) -> Plan:
    rooms = tuple(Room(r.type, tuple(fn(x, y) for x, y in r.polygon)) for r in plan.rooms)
    return dataclasses.replace(plan, rooms=rooms, doors=())


def test_canonical_key_ignores_symmetries_and_translation(real_plans):
    plan = real_plans[3]
    key = canonical_key(plan)
    assert canonical_key(_transform(plan, lambda x, y: (255 - x, y))) == key  # mirror
    assert canonical_key(_transform(plan, lambda x, y: (255 - y, x))) == key  # quarter turn
    assert canonical_key(_transform(plan, lambda x, y: (x + 3, y - 2))) == key  # shift
    assert canonical_key(_transform(plan, lambda x, y: (2 * x, y))) != key
    assert len({canonical_key(p) for p in real_plans}) == len(real_plans)


def test_deduplicate_keeps_first(real_plans):
    twin = dataclasses.replace(_transform(real_plans[0], lambda x, y: (255 - x, y)), id="twin")
    kept, removed = deduplicate([*real_plans, twin])
    assert removed == 1 and kept == real_plans


def test_split_keeps_groups_together(tmp_path):
    plans = [square_plan(f"p{k}", group=f"g{k % 37}") for k in range(400)]
    splits = split_by_group(plans, (0.8, 0.1, 0.1), seed=0)
    assert splits == split_by_group(plans, (0.8, 0.1, 0.1), seed=0)  # deterministic
    assert splits != split_by_group(plans, (0.8, 0.1, 0.1), seed=1)
    group_of = {p.id: p.group for p in plans}
    groups = {name: {group_of[i] for i in ids} for name, ids in splits.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sorted(i for ids in splits.values() for i in ids) == sorted(group_of)
    assert 0.7 < len(splits["train"]) / len(plans) < 0.9
    assert splits["val"] and splits["test"]

    write_splits(tmp_path, "toy", splits)
    assert read_split(tmp_path, "toy", "val") == splits["val"]
