"""Evaluation metrics, door re-derivation, validity verdict and the retrieval baseline."""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pytest

from conftest import square_plan
from housegen import metrics
from housegen.baselines.retrieval import RetrievalBaseline, signature
from housegen.data.schema import OUTSIDE, Door, Plan, Room
from housegen.evaluate import evaluate_method, load_eval_config, rederive_doors
from housegen.graphs import RoomGraph, graph_edit_distance
from housegen.postprocess import attach_doors, validity


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_eval_config()


def _graph(types: str, edges: list[tuple[int, int]], name: str = "g") -> RoomGraph:
    names = {"l": "living", "k": "kitchen", "b": "bedroom", "w": "bathroom", "c": "corridor"}
    return RoomGraph(
        name, tuple(names[t] for t in types), (4,) * len(types),
        tuple((i, j, "door") for i, j in edges),
    )  # fmt: skip


def _mirror(plan: Plan) -> Plan:
    rooms = tuple(
        Room(r.type, tuple((255 - x, y) for x, y in reversed(r.polygon))) for r in plan.rooms
    )
    doors = tuple(Door(d.rooms, tuple((255 - x, y) for x, y in d.segment)) for d in plan.doors)
    return dataclasses.replace(plan, id=plan.id + "_mirror", rooms=rooms, doors=doors)


# ---------------------------------------------------------------- graphs


def test_graph_edit_distance_on_known_cases():
    path = _graph("lkb", [(0, 1), (1, 2)])
    assert graph_edit_distance(path, path) == (0, True)
    assert graph_edit_distance(path, _graph("lkb", [(0, 1)])) == (1, True)  # one connection less
    assert graph_edit_distance(path, _graph("lkw", [(0, 1), (1, 2)]))[0] == 1  # one type changed
    assert graph_edit_distance(path, _graph("lk", [(0, 1)]))[0] == 2  # one room and its door
    assert graph_edit_distance(_graph("lk", [(0, 1)]), path)[0] == 2  # symmetric
    # Same rooms listed in another order: the distance ignores the numbering.
    assert graph_edit_distance(path, _graph("bkl", [(2, 1), (1, 0)]))[0] == 0
    # The two bedrooms are interchangeable: swapping them repairs both connections.
    a = _graph("lbb", [(0, 1)])
    b = _graph("lbb", [(0, 2)])
    assert graph_edit_distance(a, b)[0] == 0


def test_graph_edit_distance_matches_networkx(real_plans):
    graphs = [RoomGraph.from_plan(p) for p in real_plans[:4]]  # 4 to 7 rooms
    for a in graphs:
        for b in graphs:
            reference = nx.graph_edit_distance(
                a.to_networkx(), b.to_networkx(), node_match=lambda x, y: x["type"] == y["type"]
            )
            assert graph_edit_distance(a, b) == (reference, True)


def test_graph_edit_distance_reports_an_exhausted_budget(real_plans):
    a, b = RoomGraph.from_plan(real_plans[-1]), RoomGraph.from_plan(real_plans[-2])
    exact, proven = graph_edit_distance(a, b)
    rough, rough_proven = graph_edit_distance(a, b, max_expansions=3)
    assert proven and not rough_proven and rough >= exact


def test_room_graph_from_plan():
    graph = RoomGraph.from_plan(square_plan())
    assert graph.types == ("living", "kitchen", "bedroom", "bathroom")
    assert graph.corners == (4, 4, 4, 4) and graph.entrances == (0,)
    assert graph.edges == ((0, 1, "open"), (0, 2, "door"), (2, 3, "door"))
    assert graph.type_counts() == (1, 1, 0, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------- doors and validity


def test_real_plans_keep_their_graph_when_doors_are_rederived(real_plans, cfg):
    """Consistency check of the brief: real plans get the best possible compatibility."""
    for plan in real_plans:
        rebuilt = rederive_doors(plan, cfg)
        assert rebuilt.doors != plan.doors  # really rebuilt: doors are re-centred
        assert rebuilt.edges() == plan.edges()
        assert rebuilt.entrance_rooms() == plan.entrance_rooms()[:1]
        assert metrics.compatibility(RoomGraph.from_plan(plan), rebuilt) == (0, True)
        assert validity(rebuilt, cfg["min_area_m2"]) == "valid"


def test_attach_doors_drops_connections_the_geometry_cannot_carry():
    plan = square_plan()
    # Living room and bathroom only touch at a corner; kitchen and bedroom too.
    graph = dataclasses.replace(
        RoomGraph.from_plan(plan),
        edges=((0, 1, "open"), (0, 2, "door"), (0, 3, "door"), (1, 2, "open"), (1, 3, "door")),
    )
    rebuilt = attach_doors("x", plan.rooms, graph)
    assert rebuilt.edges() == [(0, 1, "open"), (0, 2, "door"), (1, 3, "door")]
    assert metrics.compatibility(graph, rebuilt)[0] == 2  # the two impossible connections
    door = next(d for d in rebuilt.doors if d.rooms == (0, 2))
    # 0.9 m wide, centred on the wall shared by the living room and the bedroom.
    assert sorted(door.segment) == [(115.5, 140.0), (124.5, 140.0)]
    entrance = next(d for d in rebuilt.doors if d.is_entrance)
    assert entrance.rooms == (0, OUTSIDE) and entrance.line().length == pytest.approx(10.0)


def test_attach_doors_with_a_tolerance_bridges_small_gaps():
    plan = square_plan()
    rooms = list(plan.rooms)
    rooms[2] = Room("bedroom", ((100, 141), (140, 141), (140, 180), (100, 180)))  # 10 cm gap
    graph = RoomGraph.from_plan(plan)
    assert (0, 2, "door") not in attach_doors("x", rooms, graph).edges()
    assert (0, 2, "door") in attach_doors("x", rooms, graph, tol=2.0).edges()


def test_validity_verdicts(cfg):
    plan = square_plan()
    areas = cfg["min_area_m2"]
    assert validity(plan, areas) == "valid"
    overlapping = list(plan.rooms)
    overlapping[1] = Room("kitchen", ((130, 100), (180, 100), (180, 140), (130, 140)))
    assert validity(dataclasses.replace(plan, rooms=tuple(overlapping)), areas) == "overlap"
    assert validity(dataclasses.replace(plan, doors=plan.doors[:2]), areas) == "no_entrance"
    assert validity(dataclasses.replace(plan, open_edges=()), areas) == "unreachable_room"
    tiny = list(plan.rooms)
    tiny[2] = Room("bedroom", ((100, 140), (140, 140), (140, 150), (100, 150)))  # 4 m²
    assert validity(dataclasses.replace(plan, rooms=tuple(tiny)), areas) == "implausible_area"
    twisted = list(plan.rooms)
    twisted[3] = Room("bathroom", ((140, 140), (180, 180), (180, 140), (140, 180)))
    assert validity(dataclasses.replace(plan, rooms=tuple(twisted)), areas) == "invalid_polygon"
    outside = list(plan.rooms)
    outside[1] = Room("kitchen", ((140, 100), (300, 100), (300, 140), (140, 140)))
    assert validity(dataclasses.replace(plan, rooms=tuple(outside)), areas) == "outside_grid"


# ---------------------------------------------------------------- layout similarity


def test_label_map_and_similarity():
    plan = square_plan()
    labels = metrics.label_map(plan)
    assert labels.shape == (64, 64) and labels.dtype == np.uint8
    # Four 4 m x 4 m rooms at 0.4 m per cell: 100 cells each, centred.
    assert [int((labels == k).sum()) for k in (1, 2, 4, 5)] == [100, 100, 100, 100]
    assert int((labels > 0).sum()) == 400
    assert metrics.layout_similarity(labels, labels) == 1.0
    # Translation does not matter: maps are centred on the plan.
    assert np.array_equal(labels, metrics.label_map(square_plan(shift=30)))
    # Swapping two room types changes half of the cells.
    swapped = labels.copy()
    swapped[labels == 1], swapped[labels == 2] = 2, 1
    assert metrics.layout_similarity(labels, swapped) == pytest.approx(0.5)
    assert metrics.layout_similarity(labels, np.zeros_like(labels)) == 0.0
    assert len(metrics.symmetries(labels)) == 8


def test_diversity():
    labels = metrics.label_map(square_plan())
    swapped = labels.copy()
    swapped[labels == 1], swapped[labels == 2] = 2, 1
    assert metrics.diversity([labels, labels, labels]) == 0.0
    assert metrics.diversity([labels, swapped]) == pytest.approx(0.5)
    assert np.isnan(metrics.diversity([labels]))


def test_layout_index_finds_copies_up_to_symmetry(real_plans):
    index = metrics.LayoutIndex(real_plans)
    for plan in real_plans:
        assert index.nearest(metrics.label_map(plan)) == (1.0, plan.id)
        similarity, nearest = index.nearest(metrics.label_map(_mirror(plan)))
        assert nearest == plan.id and similarity == 1.0
    # The exact search agrees with a brute-force scan over every reference and symmetry.
    query = metrics.label_map(square_plan())
    brute = max(
        metrics.layout_similarity(variant, metrics.label_map(p))
        for p in real_plans
        for variant in metrics.symmetries(query)
    )
    assert index.nearest(query)[0] == pytest.approx(brute)
    assert 0.0 < brute < 0.9  # the toy plan is not a copy of a fixture


# ---------------------------------------------------------------- areas and FID


def test_wasserstein_and_area_distance(real_plans):
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 2.0, 5000)
    assert metrics.wasserstein_1d(a, a) == 0.0
    assert metrics.wasserstein_1d(a, a + 3.0) == pytest.approx(3.0)
    assert np.isnan(metrics.wasserstein_1d(a, np.array([])))
    distances = metrics.area_distance(real_plans, real_plans)
    assert all(v == 0.0 or np.isnan(v) for v in distances.values())
    areas = metrics.room_areas([square_plan()])
    assert areas["living"].tolist() == pytest.approx([16.0]) and len(areas["dining"]) == 0


def test_frechet_distance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4000, 6))
    assert metrics.frechet_distance(a, a) == pytest.approx(0.0, abs=1e-8)
    # Shifting every feature by 2 adds |shift|^2 = 6 * 4 and leaves the covariances alone.
    assert metrics.frechet_distance(a, a + 2.0) == pytest.approx(24.0, rel=1e-6)
    # Scaling a standard normal by 3: sum over dimensions of (3 - 1)^2 = 4 each.
    b = rng.normal(size=(200_000, 3))
    c = rng.normal(size=(200_000, 3)) * 3.0
    assert metrics.frechet_distance(b, c) == pytest.approx(12.0, rel=0.03)


def test_fid_with_a_stand_in_feature_extractor(real_plans):
    def colour_histogram(images: np.ndarray) -> np.ndarray:  # no network, no weights
        return images.reshape(len(images), -1, 3).mean(axis=1).astype(float)

    images = metrics.render_for_fid(real_plans, size=64)
    assert images.shape == (7, 64, 64, 3)
    assert metrics.fid(real_plans, real_plans, colour_histogram, size=64) == pytest.approx(
        0, abs=1e-6
    )
    assert metrics.fid(real_plans, real_plans[:4], colour_histogram, size=64) > 0


# ---------------------------------------------------------------- baseline and evaluation


def test_retrieval_baseline(real_plans):
    baseline = RetrievalBaseline(real_plans, shortlist=7)
    for plan in real_plans:
        graph = RoomGraph.from_plan(plan)
        found = baseline.sample(graph, k=3)
        assert found[0].id == plan.id  # a graph of the training set retrieves its own plan
        assert len({p.id for p in found}) == 3
        assert found == baseline.sample(graph, k=3)  # deterministic
        distances = [graph_edit_distance(graph, RoomGraph.from_plan(p))[0] for p in found]
        assert distances == sorted(distances) and distances[0] == 0
    assert signature(RoomGraph.from_plan(square_plan()))[:8].tolist() == [1, 1, 0, 1, 1, 0, 0, 0]


def test_evaluate_method_on_real_and_retrieved_plans(real_plans, cfg):
    train, test = real_plans[:5], real_plans[5:]
    conditions = [RoomGraph.from_plan(p) for p in test]
    index = metrics.LayoutIndex(train)

    def features(images: np.ndarray) -> np.ndarray:
        return images.reshape(len(images), -1, 3).mean(axis=1).astype(float)

    real_features = features(metrics.render_for_fid(test, 64))
    eval_cfg = cfg | {"fid": {"image_size": 64}}
    real, rows = evaluate_method(
        "real", conditions, [[rederive_doors(p, cfg)] for p in test], test, index, eval_cfg,
        real_features, features,
    )  # fmt: skip
    assert real["compatibility_ged"] == 0.0 and real["compatible_rate"] == 1.0
    assert real["valid_rate"] == 1.0 and real["area_w1_m2"] == 0.0
    assert real["fid"] == pytest.approx(0.0, abs=1e-6) and np.isnan(real["diversity"])
    assert real["copy_rate"] == 0.0 and len(rows) == 2

    baseline = RetrievalBaseline(train, shortlist=5)
    b0, rows = evaluate_method(
        "b0", conditions, [baseline.sample(g, 3) for g in conditions], test, index, eval_cfg
    )
    assert b0["plans"] == 6 and len(rows) == 6
    assert b0["nearest_train_similarity"] == 1.0 and b0["copy_rate"] == 1.0  # nothing is new
    assert b0["compatibility_ged"] > 0 and b0["diversity"] > 0
    assert np.isnan(b0["fid"])  # no feature extractor given
    assert {r["validity"] for r in rows} == {"valid"}
