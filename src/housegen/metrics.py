"""Evaluation metrics for generated plans (section 8 of the project brief).

Every metric is a plain function of plans, so the same code scores real plans, baselines and
the diffusion model. Nothing here needs PyTorch except :class:`InceptionFeatures`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import shapely

from housegen.data.schema import ROOM_TYPES, Plan
from housegen.graphs import RoomGraph, graph_edit_distance
from housegen.viz import render_plan_image

# ---------------------------------------------------------------- compatibility


def compatibility(requested: RoomGraph, plan: Plan) -> tuple[int, bool]:
    """Edit distance between the requested graph and the graph the plan realises.

    0 means every requested room and connection is there and nothing else. Returns the
    distance and whether it is proven minimal (see :func:`graph_edit_distance`).
    """
    return graph_edit_distance(requested, RoomGraph.from_plan(plan))


# ---------------------------------------------------------------- layout similarity

MAP_SIZE = 64  # label maps are MAP_SIZE x MAP_SIZE cells
MAP_CELL = 4.0  # grid units per cell: 0.4 m with the 10 cm grid


def label_map(plan: Plan, size: int = MAP_SIZE, cell: float = MAP_CELL) -> np.ndarray:
    """Room-type raster of a plan, centred on its bounding box: 0 is empty, k + 1 is type k.

    Centring makes the map independent of where the plan sits on the canvas, and mirroring
    or turning the plan mirrors or turns the map exactly. A cell belongs to a room when its
    centre is inside the room or on its boundary; where rooms overlap, the later room wins.
    """
    drawn = [np.asarray(r.polygon, dtype=float) for r in plan.rooms if len(r.polygon) >= 3]
    if not drawn:  # every room collapsed
        return np.zeros((size, size), dtype=np.uint8)
    pts = np.concatenate(drawn)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    ticks = (np.arange(size) - (size - 1) / 2.0) * cell
    xs, ys = np.meshgrid(center[0] + ticks, center[1] + ticks)
    xs, ys = xs.ravel(), ys.ravel()
    labels = np.zeros(size * size, dtype=np.uint8)
    for room in plan.rooms:
        if len(room.polygon) >= 3:
            shape = room.shape()
            if shape.is_valid:
                labels[shapely.intersects_xy(shape, xs, ys)] = ROOM_TYPES.index(room.type) + 1
    return labels.reshape(size, size)


def layout_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the occupied cells where two label maps agree on the room type, in [0, 1].

    A type-aware intersection over union: 1 only for identical labelled layouts.
    """
    union = np.count_nonzero((a > 0) | (b > 0))
    return float(np.count_nonzero((a == b) & (a > 0)) / union) if union else 0.0


def symmetries(labels: np.ndarray) -> list[np.ndarray]:
    """The eight images of a label map under quarter turns and mirroring."""
    out = []
    for flipped in (labels, labels[:, ::-1]):
        out += [np.rot90(flipped, k) for k in range(4)]
    return out


def _pack(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bit-pack label maps (n, s, s): one-hot words (n, types, w) and occupancy words (n, w)."""
    n = len(maps)
    flat = maps.reshape(n, -1)
    onehot = np.stack([flat == k + 1 for k in range(len(ROOM_TYPES))], axis=1)
    pad = (-flat.shape[1]) % 64
    onehot = np.pad(onehot, ((0, 0), (0, 0), (0, pad)))
    words = np.packbits(onehot, axis=2).view(np.uint64)
    occupancy = np.packbits(np.pad(flat > 0, ((0, 0), (0, pad))), axis=1).view(np.uint64)
    return words, occupancy


class LayoutIndex:
    """Nearest-neighbour search among reference layouts (the training set).

    :meth:`nearest` returns the most similar reference plan under
    :func:`layout_similarity`, maximised over the eight symmetries of the query: a rotated or
    mirrored copy of a training plan is still a copy. The search is exact; it uses the fact
    that the similarity can never exceed the plain footprint IoU to skip most references.
    """

    def __init__(self, plans: Sequence[Plan]) -> None:
        self.ids = [p.id for p in plans]
        maps = np.stack([label_map(p) for p in plans])
        self.onehot, self.occupancy = _pack(maps)
        self.area = np.bitwise_count(self.occupancy).sum(axis=1).astype(np.int64)

    def nearest(self, labels: np.ndarray, chunk: int = 256) -> tuple[float, str]:
        best, best_id = -1.0, ""
        for variant in symmetries(labels):
            onehot, occupancy = _pack(variant[None])
            area = int(np.bitwise_count(occupancy).sum())
            both = np.bitwise_count(self.occupancy & occupancy).sum(axis=1).astype(np.int64)
            union = self.area + area - both
            bound = both / np.maximum(union, 1)  # footprint IoU >= typed similarity
            order = np.argsort(-bound, kind="stable")
            for start in range(0, len(order), chunk):
                idx = order[start : start + chunk]
                if bound[idx[0]] <= best:
                    break
                agree = np.bitwise_count(self.onehot[idx] & onehot).sum(axis=(1, 2))
                sim = agree / np.maximum(union[idx], 1)
                k = int(np.argmax(sim))
                if sim[k] > best:
                    best, best_id = float(sim[k]), self.ids[int(idx[k])]
        return best, best_id


def diversity(maps: Sequence[np.ndarray]) -> float:
    """Mean of ``1 - similarity`` over all pairs of plans generated for one graph."""
    pairs = [
        1.0 - layout_similarity(maps[i], maps[j])
        for i in range(len(maps))
        for j in range(i + 1, len(maps))
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


# ---------------------------------------------------------------- areas


def room_areas(plans: Sequence[Plan]) -> dict[str, np.ndarray]:
    """Room areas in square metres, grouped by room type."""
    areas: dict[str, list[float]] = {t: [] for t in ROOM_TYPES}
    for plan in plans:
        for i, room in enumerate(plan.rooms):
            if len(room.polygon) >= 3:
                areas[room.type].append(plan.room_area_m2(i))
    return {t: np.asarray(v) for t, v in areas.items()}


def wasserstein_1d(a: np.ndarray, b: np.ndarray, quantiles: int = 200) -> float:
    """1-Wasserstein distance between two samples, from their quantile functions."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def area_distance(real: Sequence[Plan], generated: Sequence[Plan]) -> dict[str, float]:
    """Per room type, the Wasserstein distance (m²) between real and generated room areas."""
    a, b = room_areas(real), room_areas(generated)
    return {t: wasserstein_1d(a[t], b[t]) for t in ROOM_TYPES}


# ---------------------------------------------------------------- FID


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets (rows are samples).

    ``|mu_a - mu_b|^2 + tr(S_a + S_b - 2 (S_a S_b)^(1/2))``; the trace of the matrix square
    root is the sum of the square roots of the eigenvalues of ``S_a S_b``.
    """
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(b, rowvar=False))
    eigenvalues = np.linalg.eigvals(cov_a @ cov_b)
    trace_sqrt = np.sqrt(np.clip(eigenvalues.real, 0.0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_sqrt)


def render_for_fid(plans: Sequence[Plan], size: int = 256) -> np.ndarray:
    """Fixed-style renders (n, size, size, 3), without doors so raw model output is comparable."""
    return np.stack([render_plan_image(p, size=size, doors=False) for p in plans])


class InceptionFeatures:
    """Pool3 features of the FID Inception network (``pytorch-fid`` weights), in batches.

    Needs the ``cpu`` or ``cu126`` extra. The weights (about 90 MB) are downloaded by
    ``pytorch-fid`` on first use and cached by PyTorch.
    """

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        import torch
        from pytorch_fid.inception import InceptionV3

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block]).to(self.device).eval()

    def __call__(self, images: np.ndarray) -> np.ndarray:
        torch, out = self.torch, []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(images[start : start + self.batch_size])
                batch = batch.to(self.device).permute(0, 3, 1, 2).float() / 255.0
                out.append(self.model(batch)[0].flatten(1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def fid(
    real: Sequence[Plan],
    generated: Sequence[Plan],
    features: Callable[[np.ndarray], np.ndarray],
    size: int = 256,
) -> float:
    """FID between renders of real and generated plans, for a given feature extractor."""
    return frechet_distance(
        features(render_for_fid(real, size)), features(render_for_fid(generated, size))
    )
