"""Plans and room graphs as tensors for the corner diffusion model.

A plan is a set of corners, one token each. The k-th token of a room is its k-th corner,
rings being counter-clockwise and starting at the lowest, then leftmost, corner, so the
target of every token is well defined. Each token carries the conditioning read from the
room graph (room type, corner index, corner count of its room, room identity, entrance flag);
the graph's connections become the relational attention mask (:mod:`housegen.model.masks`).

Training augmentations are cheap tensor operations: the eight symmetries of the square
(precomputed per plan) and a random relabelling of the room identities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch

from housegen.data.schema import GRID_SIZE, ROOM_TYPES, Plan
from housegen.graphs import RoomGraph

MAX_ROOMS = 16  # room identity slots; filtered plans have at most 10 rooms
MAX_CORNERS = 12  # corners per room
CENTER = (GRID_SIZE - 1) / 2.0  # grid coordinate mapped to 0


def to_unit(points):
    """Grid coordinates (0..255) to model coordinates (-1..1)."""
    return (points - CENTER) / CENTER


def to_grid(x):
    """Model coordinates (-1..1) to grid coordinates (0..255), not rounded."""
    return x * CENTER + CENTER


def canonical_ring(points: np.ndarray) -> np.ndarray:
    """The same ring, counter-clockwise, starting at its lowest then leftmost corner."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pts = pts[::-1]
    start = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    return np.roll(pts, -start, axis=0)


def d4_matrices() -> np.ndarray:
    """The eight symmetries of the square as 2 x 2 integer matrices, identity first."""
    rotation = np.array([[0, -1], [1, 0]])
    mirror = np.array([[-1, 0], [0, 1]])
    out = []
    for m in (np.eye(2, dtype=int), mirror):
        r = np.eye(2, dtype=int)
        for _ in range(4):
            out.append(r @ m)
            r = rotation @ r
    return np.stack(out)


D4 = d4_matrices()


def graph_tokens(graph: RoomGraph) -> dict[str, np.ndarray]:
    """Per-token conditioning of a room graph, and its room adjacency matrix."""
    if len(graph.types) > MAX_ROOMS:
        raise ValueError(f"at most {MAX_ROOMS} rooms are supported")
    room, kind, corner, count, entrance, nxt = [], [], [], [], [], []
    for i, (room_type, n) in enumerate(zip(graph.types, graph.corners, strict=True)):
        if not 3 <= n <= MAX_CORNERS:
            raise ValueError(f"room {i} has {n} corners, supported: 3 to {MAX_CORNERS}")
        start = len(room)
        room += [i] * n
        kind += [ROOM_TYPES.index(room_type)] * n
        corner += list(range(n))
        count += [n] * n
        entrance += [int(i in graph.entrances)] * n
        nxt += [start + (k + 1) % n for k in range(n)]
    adjacency = np.zeros((MAX_ROOMS, MAX_ROOMS), dtype=bool)
    for i, j, _ in graph.edges:
        adjacency[i, j] = adjacency[j, i] = True
    tokens = {"room": room, "type": kind, "corner": corner, "count": count,
              "entrance": entrance, "next": nxt}  # fmt: skip
    out = {key: np.asarray(value, dtype=np.int64) for key, value in tokens.items()}
    out["adjacency"] = adjacency
    return out


def plan_corners(plan: Plan) -> np.ndarray:
    """Corner coordinates of ``plan`` in model units, one copy per symmetry: (8, L, 2)."""
    variants = []
    for m in D4:
        rings = [
            canonical_ring((np.asarray(r.polygon, dtype=float) - CENTER) @ m.T + CENTER)
            for r in plan.rooms
        ]
        variants.append(to_unit(np.concatenate(rings)))
    return np.stack(variants).astype(np.float32)


@dataclass
class Batch:
    """A padded batch of plans (or graphs). Padding tokens have ``valid == False``."""

    x0: torch.Tensor  # (B, L, 2) corners in model units (zeros when only graphs are known)
    room: torch.Tensor  # (B, L) room index in the plan
    room_id: torch.Tensor  # (B, L) room identity seen by the model
    type: torch.Tensor  # (B, L) room type index
    corner: torch.Tensor  # (B, L) corner index in the room
    count: torch.Tensor  # (B, L) corner count of the room
    entrance: torch.Tensor  # (B, L) 1 for the corners of an entrance room
    next: torch.Tensor  # (B, L) token index of the next corner of the same room
    valid: torch.Tensor  # (B, L) bool
    adjacency: torch.Tensor  # (B, MAX_ROOMS, MAX_ROOMS) bool, rooms connected in the graph

    def to(self, device) -> Batch:
        return Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class PlanTensors:
    """Encoded plans (or graphs) stored on one device, ready to be batched."""

    def __init__(
        self,
        graphs: Sequence[RoomGraph],
        corners: Sequence[np.ndarray] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.ids = [g.id for g in graphs]
        self.graphs = list(graphs)
        tokens = [graph_tokens(g) for g in graphs]
        lengths = np.array([len(t["room"]) for t in tokens])
        n, length = len(tokens), int(lengths.max())

        def pad(key: str) -> torch.Tensor:
            arr = np.zeros((n, length), dtype=np.int64)
            for k, t in enumerate(tokens):
                arr[k, : len(t[key])] = t[key]
            return torch.from_numpy(arr).to(self.device)

        self.room, self.type, self.corner = pad("room"), pad("type"), pad("corner")
        self.count, self.entrance, self.next = pad("count"), pad("entrance"), pad("next")
        self.lengths = torch.from_numpy(lengths).to(self.device)
        self.valid = torch.arange(length, device=self.device)[None, :] < self.lengths[:, None]
        self.adjacency = torch.from_numpy(np.stack([t["adjacency"] for t in tokens])).to(
            self.device
        )
        coords = np.zeros((n, len(D4), length, 2), dtype=np.float32)
        if corners is not None:
            for k, c in enumerate(corners):
                coords[k, :, : c.shape[1]] = c
        self.coords = torch.from_numpy(coords).to(self.device)

    @classmethod
    def from_plans(cls, plans: Sequence[Plan], device: str | torch.device = "cpu") -> PlanTensors:
        return cls(
            [RoomGraph.from_plan(p) for p in plans], [plan_corners(p) for p in plans], device
        )

    @classmethod
    def from_graphs(
        cls, graphs: Sequence[RoomGraph], device: str | torch.device = "cpu"
    ) -> PlanTensors:
        return cls(graphs, None, device)

    def __len__(self) -> int:
        return len(self.ids)

    def batch(
        self,
        index: torch.Tensor,
        generator: torch.Generator | None = None,
        d4: bool = False,
        room_ids: bool = False,
    ) -> Batch:
        """Plans ``index``, padded to their longest one.

        ``d4`` applies a random symmetry of the square to each plan; ``room_ids`` gives the
        rooms random identities among ``MAX_ROOMS`` slots instead of their index.
        """
        index = index.to(self.device)
        size = len(index)
        length = int(self.lengths[index].max())
        if d4:
            variant = torch.randint(0, len(D4), (size,), generator=generator, device=self.device)
        else:
            variant = torch.zeros(size, dtype=torch.long, device=self.device)
        room = self.room[index, :length]
        if room_ids:
            noise = torch.rand(size, MAX_ROOMS, generator=generator, device=self.device)
            room_id = torch.argsort(noise, dim=1).gather(1, room)
        else:
            room_id = room
        return Batch(
            x0=self.coords[index, variant, :length],
            room=room,
            room_id=room_id,
            type=self.type[index, :length],
            corner=self.corner[index, :length],
            count=self.count[index, :length],
            entrance=self.entrance[index, :length],
            next=self.next[index, :length],
            valid=self.valid[index, :length],
            adjacency=self.adjacency[index],
        )


class BatchSchedule:
    """Deterministic order of training batches: batch ``m`` depends only on ``(seed, m)``.

    Each epoch is a seeded permutation; within chunks of 50 batches plans are sorted by
    length so that a batch holds plans of similar sizes (less padding), then the batches
    are shuffled. Resuming at batch ``m`` needs no saved sampler state.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, chunk: int = 50) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = min(batch_size, len(self.lengths))
        self.per_epoch = len(self.lengths) // self.batch_size
        self.seed, self.chunk = seed, chunk
        self._epoch, self._batches = -1, []

    def _make(self, epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng([self.seed, epoch])
        used = self.per_epoch * self.batch_size
        perm = rng.permutation(len(self.lengths))[:used]
        span = self.batch_size * self.chunk
        batches = []
        for s in range(0, used, span):
            part = perm[s : s + span]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[k : k + self.batch_size] for k in range(0, len(part), self.batch_size)]
        return [batches[k] for k in rng.permutation(len(batches))]

    def __call__(self, m: int) -> np.ndarray:
        epoch, position = divmod(m, self.per_epoch)
        if epoch != self._epoch:
            self._epoch, self._batches = epoch, self._make(epoch)
        return self._batches[position]
