"""B0: retrieval baseline. Answers a graph with the training plans whose graph is closest.

Retrieved plans are real, so they are as realistic as plans get, but nothing is invented:
this baseline anchors the novelty metric (its nearest-training-plan similarity is 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations_with_replacement

import numpy as np

from housegen.data.schema import ROOM_TYPES, Plan
from housegen.graphs import RoomGraph, graph_edit_distance

_PAIRS = {
    pair: k for k, pair in enumerate(combinations_with_replacement(range(len(ROOM_TYPES)), 2))
}


def signature(graph: RoomGraph) -> np.ndarray:
    """Cheap graph descriptor: room count per type, then connection count per pair of types."""
    edges = np.zeros(len(_PAIRS), dtype=np.int32)
    for i, j, _ in graph.edges:
        a, b = sorted((ROOM_TYPES.index(graph.types[i]), ROOM_TYPES.index(graph.types[j])))
        edges[_PAIRS[(a, b)]] += 1
    return np.concatenate([np.asarray(graph.type_counts(), dtype=np.int32), edges])


class RetrievalBaseline:
    """Nearest training graphs: a signature shortlist re-ranked by exact edit distance."""

    def __init__(self, train: Sequence[Plan], shortlist: int = 30) -> None:
        self.plans = sorted(train, key=lambda p: p.id)
        self.graphs = [RoomGraph.from_plan(p) for p in self.plans]
        self.signatures = np.stack([signature(g) for g in self.graphs])
        self.shortlist = shortlist
        # A missing room weighs more than a missing connection.
        self.weights = np.array([3] * len(ROOM_TYPES) + [1] * len(_PAIRS))

    def sample(self, graph: RoomGraph, k: int = 5) -> list[Plan]:
        """The ``k`` training plans closest to ``graph`` (deterministic)."""
        score = (np.abs(self.signatures - signature(graph)) * self.weights).sum(axis=1)
        shortlist = np.argsort(score, kind="stable")[: max(self.shortlist, k)]
        ranked = sorted(
            shortlist, key=lambda i: (graph_edit_distance(graph, self.graphs[i])[0], score[i], i)
        )
        return [self.plans[i] for i in ranked[:k]]
