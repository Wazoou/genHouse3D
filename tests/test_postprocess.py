"""Geometric clean-up of generated plans: every step on hand-made defects."""

from __future__ import annotations

import dataclasses

import pytest
from shapely.geometry import Polygon

from conftest import square_plan
from housegen.data.schema import Room
from housegen.graphs import RoomGraph
from housegen.metrics import compatibility
from housegen.postprocess import CleanupConfig, clean_plan, requested_graph, validity

# The toy plan: living (100..140, 100..140), kitchen (140..180, 100..140),
# bedroom (100..140, 140..180), bathroom (140..180, 140..180); living-kitchen open,
# living-bedroom and bedroom-bathroom doors, entrance on the living room.


def _with_rooms(*polygons):
    plan = square_plan()
    rooms = tuple(Room(r.type, p) for r, p in zip(plan.rooms, polygons, strict=True))
    return dataclasses.replace(plan, rooms=rooms, doors=(), open_edges=())


def _clean(plan):
    return clean_plan(plan, RoomGraph.from_plan(square_plan()))


def test_clean_plans_are_left_alone(real_plans):
    for plan in [square_plan(), *real_plans]:
        stripped = dataclasses.replace(plan, doors=(), open_edges=())
        cleaned, stats = clean_plan(stripped, RoomGraph.from_plan(plan))
        assert stats == dict.fromkeys(
            ("repaired", "not_rectilinear", "cropped", "enclosed", "collapsed"), 0
        )
        for got, want in zip(cleaned.rooms, plan.rooms, strict=True):
            assert Polygon(got.polygon).equals(want.shape())
        assert cleaned.edges() == plan.edges()
        assert compatibility(RoomGraph.from_plan(plan), cleaned)[0] == 0


def test_gaps_and_small_overlaps_are_closed():
    plan = _with_rooms(
        ((100, 100), (138, 100), (138, 141), (100, 141)),  # 20 cm short of the kitchen
        ((141, 100), (180, 100), (180, 139), (141, 139)),
        ((100, 143), (139, 143), (139, 180), (100, 180)),  # 20 cm gap above the living room
        ((137, 138), (180, 138), (180, 180), (137, 180)),  # overlaps two neighbours by 0.2 m
    )
    cleaned, stats = _clean(plan)
    assert stats["cropped"] == 0  # the overlaps were small: walls moved, nothing was cut
    shapes = [r.shape() for r in cleaned.rooms]
    assert all(s.is_valid for s in shapes)
    # The gaps are closed: the rooms tile their bounding box without holes.
    union = shapes[0].union(shapes[1]).union(shapes[2]).union(shapes[3])
    assert union.area == pytest.approx(union.envelope.area)
    for i in range(4):
        for j in range(i + 1, 4):
            assert shapes[i].intersection(shapes[j]).area == 0
    assert cleaned.edges() == [(0, 1, "open"), (0, 2, "door"), (2, 3, "door")]
    assert cleaned.entrance_rooms() == [0]
    assert validity(cleaned) == "valid"
    assert compatibility(RoomGraph.from_plan(square_plan()), cleaned)[0] == 0


def test_slanted_walls_are_straightened():
    plan = _with_rooms(
        ((100, 100), (140, 101), (139, 140), (101, 139)),  # every wall tilted by about 1.5°
        ((140, 100), (180, 100), (180, 140), (140, 140)),
        ((100, 140), (140, 140), (140, 180), (100, 180)),
        ((140, 140), (180, 140), (180, 180), (140, 180)),
    )
    cleaned, _ = _clean(plan)
    for room in cleaned.rooms:
        xs = {x for x, _ in room.polygon}
        ys = {y for _, y in room.polygon}
        assert len(room.polygon) == 4 and len(xs) == 2 and len(ys) == 2  # axis-aligned box
    assert validity(cleaned) == "valid"


def test_large_overlap_crops_the_bigger_room():
    plan = _with_rooms(
        ((100, 100), (140, 100), (140, 140), (100, 140)),
        ((140, 100), (180, 100), (180, 140), (140, 140)),
        ((100, 140), (180, 140), (180, 180), (100, 180)),  # bedroom covers the bathroom...
        ((150, 150), (180, 150), (180, 180), (150, 180)),  # ...a small room in its corner
    )
    cleaned, stats = _clean(plan)
    bedroom, bathroom = cleaned.rooms[2].shape(), cleaned.rooms[3].shape()
    assert stats["cropped"] == 1
    assert bathroom.equals(Polygon(plan.rooms[3].polygon))  # the smaller room keeps its shape
    assert bedroom.intersection(bathroom).area == 0
    assert bedroom.area == 80 * 40 - 30 * 30  # the bigger one is cropped into an L
    assert validity(cleaned) == "valid"


def test_a_room_inside_another_stays_an_overlap():
    """A room fully inside another would leave a hole, which the format cannot hold."""
    plan = _with_rooms(
        ((100, 100), (140, 100), (140, 140), (100, 140)),
        ((140, 100), (180, 100), (180, 140), (140, 140)),
        ((100, 140), (180, 140), (180, 180), (100, 180)),
        ((150, 150), (170, 150), (170, 170), (150, 170)),
    )
    cleaned, stats = _clean(plan)
    assert stats["enclosed"] == 1
    assert validity(cleaned) == "overlap"


def test_self_intersecting_room_is_repaired():
    plan = _with_rooms(
        ((100, 100), (140, 100), (140, 140), (100, 140)),
        ((140, 100), (180, 140), (180, 100), (140, 140)),  # bow tie
        ((100, 140), (140, 140), (140, 180), (100, 180)),
        ((140, 140), (180, 140), (180, 180), (140, 180)),
    )
    cleaned, stats = _clean(plan)
    assert stats["repaired"] == 1
    assert all(r.shape().is_valid for r in cleaned.rooms if len(r.polygon) >= 3)


def test_collapsed_room_makes_the_plan_invalid():
    plan = _with_rooms(
        ((100, 100), (140, 100), (140, 140), (100, 140)),
        ((140, 100), (180, 100), (180, 140), (140, 140)),
        ((100, 140), (140, 140), (140, 180), (100, 180)),
        ((140, 140), (140, 140), (140, 140), (140, 140)),  # all corners on one point
    )
    cleaned, stats = _clean(plan)
    assert stats["collapsed"] == 1 and cleaned.rooms[3].polygon == ()
    assert validity(cleaned) == "invalid_polygon"


def test_entrance_falls_back_to_the_living_room():
    plan = square_plan()
    graph = dataclasses.replace(RoomGraph.from_plan(plan), entrances=(3,))
    # The bathroom has outer walls, so it gets the entrance as requested...
    cleaned, _ = clean_plan(dataclasses.replace(plan, doors=()), graph)
    assert cleaned.entrance_rooms() == [3]
    # ...but a room enclosed by the others cannot: the living room takes it.
    enclosed = _with_rooms(
        ((100, 100), (180, 100), (180, 120), (100, 120)),  # living: bottom strip
        ((100, 120), (120, 120), (120, 160), (100, 160)),  # kitchen: left
        ((100, 160), (160, 160), (160, 120), (180, 120), (180, 180), (100, 180)),  # bedroom: L
        ((120, 120), (160, 120), (160, 160), (120, 160)),  # bathroom: in the middle
    )
    inner_graph = dataclasses.replace(
        graph, edges=((0, 1, "door"), (1, 2, "door"), (2, 3, "door")), entrances=(3,)
    )
    cleaned, _ = clean_plan(enclosed, inner_graph)
    assert cleaned.entrance_rooms() == [0]


def test_requested_graph_round_trip():
    plan = square_plan()
    graph = RoomGraph.from_plan(plan)
    meta = {"condition": "q", "edges": [list(e) for e in graph.edges], "entrances": [0]}
    restored = requested_graph(dataclasses.replace(plan, doors=(), meta=meta))
    assert restored == dataclasses.replace(graph, id="q")


def test_thresholds_are_configurable():
    plan = _with_rooms(
        ((100, 100), (136, 100), (136, 140), (100, 140)),  # 40 cm gap to the kitchen
        ((140, 100), (180, 100), (180, 140), (140, 140)),
        ((100, 140), (136, 140), (136, 180), (100, 180)),
        ((140, 140), (180, 140), (180, 180), (140, 180)),
    )
    graph = RoomGraph.from_plan(square_plan())
    tight, _ = clean_plan(plan, graph, CleanupConfig(max_gap_m=0.2))
    loose, _ = clean_plan(plan, graph, CleanupConfig(max_gap_m=0.5))
    assert (0, 1, "open") not in tight.edges()
    assert (0, 1, "open") in loose.edges()
