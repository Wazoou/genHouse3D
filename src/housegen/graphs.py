"""Room graphs (bubble diagrams): the conditioning of the generator, and how to compare them."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from housegen.data.schema import ROOM_TYPES, Plan


@dataclass(frozen=True)
class RoomGraph:
    """What the user asks for: typed rooms, their corner counts and their connections."""

    id: str
    types: tuple[str, ...]
    corners: tuple[int, ...]  # number of corners of each room
    edges: tuple[tuple[int, int, str], ...]  # (i, j, kind) with i < j, kind in {"door", "open"}
    entrances: tuple[int, ...] = ()  # rooms carrying an entrance door

    @classmethod
    def from_plan(cls, plan: Plan) -> RoomGraph:
        return cls(
            id=plan.id,
            types=tuple(r.type for r in plan.rooms),
            corners=tuple(len(r.polygon) for r in plan.rooms),
            edges=tuple(plan.edges()),
            entrances=tuple(plan.entrance_rooms()),
        )

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        for i, room_type in enumerate(self.types):
            g.add_node(i, type=room_type, corners=self.corners[i])
        for i, j, kind in self.edges:
            g.add_edge(i, j, kind=kind)
        return g

    def type_counts(self) -> tuple[int, ...]:
        return tuple(self.types.count(t) for t in ROOM_TYPES)


def graph_edit_distance(
    a: RoomGraph, b: RoomGraph, max_expansions: int = 200_000
) -> tuple[int, bool]:
    """Edit distance between two room graphs, and whether it is proven minimal.

    Unit costs: deleting or inserting a room, changing a room type, deleting or inserting a
    connection (the kind of connection is ignored). Exact branch and bound over the room
    mappings, written for these small graphs (at most about ten rooms); the search stops
    after ``max_expansions`` steps and then returns the best distance found, flagged as not
    proven.
    """
    n1, n2 = len(a.types), len(b.types)
    adj1, adj2 = [0] * n1, [0] * n2
    for i, j, _ in a.edges:
        adj1[i] |= 1 << j
        adj1[j] |= 1 << i
    for i, j, _ in b.edges:
        adj2[i] |= 1 << j
        adj2[j] |= 1 << i
    edges2 = [(i, j) for i, j, _ in b.edges]
    order = sorted(range(n1), key=lambda u: -adj1[u].bit_count())  # constrained rooms first
    # Best known solution: the identity mapping when the rooms are the same, otherwise
    # deleting everything in ``a`` and inserting everything in ``b``.
    if a.types == b.types:
        best = len({(i, j) for i, j, _ in a.edges} ^ set(edges2))
    else:
        best = n1 + n2 + len(a.edges) + len(edges2)
    budget = max_expansions
    mapping = [-1] * n1  # image in ``b`` of each room of ``a``, -1 when deleted

    def open_edges_b(used: int) -> int:
        """Connections of ``b`` with at least one room not matched yet."""
        return sum(1 for i, j in edges2 if not (used >> i & 1 and used >> j & 1))

    def lower_bound(k: int, used: int) -> int:
        rest = order[k:]
        rest_mask = sum(1 << u for u in rest)
        types1 = [a.types[u] for u in rest]
        types2 = [b.types[v] for v in range(n2) if not used >> v & 1]
        common = sum(min(types1.count(t), types2.count(t)) for t in set(types1))
        # Connections of ``a`` not settled yet: those touching a room still to be mapped.
        inner = sum((adj1[u] & rest_mask).bit_count() for u in rest) // 2
        outer = sum((adj1[u] & ~rest_mask).bit_count() for u in rest)
        return max(len(types1), len(types2)) - common + abs(inner + outer - open_edges_b(used))

    def search(k: int, used: int, cost: int) -> None:
        nonlocal best, budget
        if budget <= 0 or cost + lower_bound(k, used) >= best:
            return
        budget -= 1
        if k == n1:  # every remaining room and connection of ``b`` is inserted
            best = min(best, cost + n2 - used.bit_count() + open_edges_b(used))
            return
        u, earlier = order[k], order[:k]
        free = [v for v in range(n2) if not used >> v & 1]
        for v in sorted(free, key=lambda v: b.types[v] != a.types[u]):
            delta = int(b.types[v] != a.types[u])
            for w in earlier:
                in_a = bool(adj1[u] >> w & 1)
                in_b = mapping[w] >= 0 and bool(adj2[v] >> mapping[w] & 1)
                delta += in_a != in_b
            mapping[u] = v
            search(k + 1, used | 1 << v, cost + delta)
        mapping[u] = -1
        search(k + 1, used, cost + 1 + sum(adj1[u] >> w & 1 for w in earlier))

    search(0, 0, 0)
    return best, budget > 0


# ---------------------------------------------------------------- graph selection (V1)


def has_open_kitchen(graph: RoomGraph) -> bool:
    """True when a kitchen opens onto another room without a wall."""
    kitchens = {i for i, t in enumerate(graph.types) if t == "kitchen"}
    return any(kind == "open" and (i in kitchens or j in kitchens) for i, j, kind in graph.edges)


def select_graph(
    graphs: list[RoomGraph],
    bedrooms: int,
    bathrooms: int,
    open_kitchen: bool | None = None,
    seed: int = 0,
) -> tuple[RoomGraph, bool]:
    """A real graph matching the request, drawn with ``seed``; and whether it matches exactly.

    The user does not draw a graph: they ask for room counts and get a real graph from the
    test split that has them (the model then generates a new plan for it). When no graph
    matches, the closest request is used: bedrooms first, then bathrooms, then the kitchen.
    """
    import random

    def gap(g: RoomGraph) -> tuple[int, int, int]:
        kitchen = 0 if open_kitchen is None else int(has_open_kitchen(g) != open_kitchen)
        return (abs(g.types.count("bedroom") - bedrooms),
                abs(g.types.count("bathroom") - bathrooms), kitchen)  # fmt: skip

    best = min(gap(g) for g in graphs)
    candidates = sorted((g for g in graphs if gap(g) == best), key=lambda g: g.id)
    return random.Random(seed).choice(candidates), best == (0, 0, 0)
