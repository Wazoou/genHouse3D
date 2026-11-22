"""Corner encoding, transformer, diffusion, EMA and sampling (CPU, tiny models)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shapely.geometry import Polygon  # noqa: E402

from conftest import square_plan  # noqa: E402
from housegen.data.schema import GRID_SIZE  # noqa: E402
from housegen.graphs import RoomGraph  # noqa: E402
from housegen.model.diffusion import Diffusion, cosine_betas  # noqa: E402
from housegen.model.ema import EMA  # noqa: E402
from housegen.model.encoding import (  # noqa: E402
    D4,
    BatchSchedule,
    PlanTensors,
    canonical_ring,
    plan_corners,
    to_grid,
    to_unit,
)
from housegen.model.transformer import CornerTransformer  # noqa: E402
from housegen.sample import decode_rooms, generate  # noqa: E402


def tiny_model(**kwargs) -> CornerTransformer:
    torch.manual_seed(0)
    return CornerTransformer(**({"dim": 32, "depth": 2, "heads": 4, "fourier": 2} | kwargs))


# ---------------------------------------------------------------- encoding


def test_canonical_ring_and_symmetries():
    ring = canonical_ring(np.array([(4, 0), (4, 3), (0, 3), (0, 0)]))
    assert ring.tolist() == [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert canonical_ring(ring[::-1]).tolist() == ring.tolist()  # clockwise input
    assert len(D4) == 8 and np.array_equal(D4[0], np.eye(2))
    assert len({m.tobytes() for m in D4}) == 8
    assert all(abs(round(np.linalg.det(m))) == 1 for m in D4)


def test_plan_corners_round_trip(real_plans):
    for plan in real_plans:
        corners = plan_corners(plan)
        assert corners.shape == (8, sum(len(r.polygon) for r in plan.rooms), 2)
        assert np.abs(corners).max() <= 1.0
        graph = RoomGraph.from_plan(plan)
        for variant in corners:
            rooms = decode_rooms(variant, graph)
            areas = [Polygon(r.polygon).area for r in rooms]
            assert areas == pytest.approx([r.shape().area for r in plan.rooms])
            assert all(Polygon(r.polygon).exterior.is_ccw for r in rooms)
        identity = decode_rooms(corners[0], graph)  # same rooms, canonical start corner
        for got, want in zip(identity, plan.rooms, strict=True):
            assert set(got.polygon) == set(want.polygon) and got.type == want.type
    assert to_grid(to_unit(np.array([0.0, 127.5, 255.0]))).tolist() == [0.0, 127.5, 255.0]


def test_plan_tensors_and_batch(real_plans):
    data = PlanTensors.from_plans(real_plans)
    lengths = [sum(len(r.polygon) for r in p.rooms) for p in real_plans]
    assert data.lengths.tolist() == lengths and len(data) == 7
    batch = data.batch(torch.tensor([0, 6]))
    assert batch.x0.shape == (2, max(lengths[0], lengths[6]), 2)
    assert batch.valid.sum(1).tolist() == [lengths[0], lengths[6]]
    assert torch.equal(batch.room_id, batch.room)  # no relabelling asked
    # "next" points to the following corner of the same room, wrapping around.
    k = 0
    for i, room in enumerate(real_plans[0].rooms):
        n = len(room.polygon)
        assert batch.next[0, k : k + n].tolist() == [k + (j + 1) % n for j in range(n)]
        assert batch.room[0, k : k + n].tolist() == [i] * n
        assert batch.count[0, k : k + n].tolist() == [n] * n
        k += n
    g = torch.Generator().manual_seed(0)
    augmented = data.batch(torch.tensor([0, 6]), g, d4=True, room_ids=True)
    for b in range(2):
        valid = augmented.valid[b]
        ids = augmented.room_id[b][valid]
        rooms = augmented.room[b][valid]
        # Room identities are a relabelling: one identity per room, shared by its corners.
        assert len(set(zip(rooms.tolist(), ids.tolist(), strict=True))) == len(set(rooms.tolist()))
        assert len(set(ids.tolist())) == len(set(rooms.tolist()))


def test_graph_only_tensors():
    graph = RoomGraph.from_plan(square_plan())
    batch = PlanTensors.from_graphs([graph]).batch(torch.arange(1))
    assert batch.x0.shape == (1, 16, 2) and not batch.x0.any()
    assert batch.adjacency[0, 0, 2] and batch.adjacency[0, 2, 0] and not batch.adjacency[0, 1, 3]
    assert batch.entrance[0].tolist() == [1] * 4 + [0] * 12
    with pytest.raises(ValueError, match="corners"):
        PlanTensors.from_graphs([RoomGraph("x", ("living",), (14,), ())])


def test_batch_schedule_is_deterministic_and_covers_every_plan():
    lengths = np.random.default_rng(0).integers(20, 70, size=103)
    schedule = BatchSchedule(lengths, batch_size=10, seed=3)
    assert schedule.per_epoch == 10
    epoch = [schedule(m) for m in range(10)]
    seen = np.concatenate(epoch)
    assert len(seen) == 100 and len(set(seen.tolist())) == 100
    again = BatchSchedule(lengths, batch_size=10, seed=3)
    assert all(np.array_equal(again(m), batch) for m, batch in enumerate(epoch))
    assert not np.array_equal(BatchSchedule(lengths, 10, seed=4)(0), epoch[0])
    assert not np.array_equal(schedule(10), epoch[0])  # the next epoch is reshuffled
    # Within a batch, lengths are close: plans are bucketed by size.
    spread = np.mean([np.ptp(lengths[b]) for b in epoch])
    assert spread < np.ptp(lengths) / 2


# ---------------------------------------------------------------- transformer


def test_transformer_shapes_and_zero_init(real_plans):
    model = tiny_model()
    batch = PlanTensors.from_plans(real_plans).batch(torch.arange(7))
    out = model(batch.x0, torch.full((7,), 10), batch)
    assert out.shape == batch.x0.shape
    assert torch.equal(out, torch.zeros_like(out))  # adaLN-Zero: the network starts at 0
    with pytest.raises(ValueError, match="attentions"):
        CornerTransformer(attentions=["doors"])


def test_outputs_do_not_depend_on_padding(real_plans):
    model = tiny_model()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    data = PlanTensors.from_plans(real_plans)
    alone = data.batch(torch.tensor([0]))
    padded = data.batch(torch.tensor([0, 6]))  # plan 6 is longer: plan 0 gets padding
    n = int(alone.valid.sum())
    x = torch.rand(1, n, 2) * 2 - 1
    x_padded = torch.cat([x, torch.rand(1, padded.x0.shape[1] - n, 2)], dim=1)
    x_padded = torch.cat([x_padded, torch.rand_like(x_padded)])
    t = torch.tensor([300, 300])
    out_alone = model(x, t[:1], alone)
    out_padded = model(x_padded, t, padded)
    assert torch.allclose(out_alone[0], out_padded[0, :n], atol=1e-5)


def test_outputs_follow_a_relabelling_of_rooms(real_plans):
    """Room identities are arbitrary labels: the model is equivariant to token order."""
    model = tiny_model()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    batch = PlanTensors.from_plans(real_plans).batch(torch.tensor([3]))
    x, t = batch.x0 + 0.05 * torch.randn_like(batch.x0), torch.tensor([200])
    perm = torch.randperm(x.shape[1])
    inverse = torch.argsort(perm)
    shuffled = type(batch)(**{
        k: (v[:, perm] if v.dim() == 2 else v[:, perm] if k == "x0" else v)
        for k, v in batch.__dict__.items()
    })  # fmt: skip
    shuffled.next = inverse[batch.next[:, perm]]
    assert torch.allclose(model(x, t, batch)[:, perm], model(x[:, perm], t, shuffled), atol=1e-5)


# ---------------------------------------------------------------- diffusion


def test_cosine_schedule():
    betas = cosine_betas(1000)
    assert betas.shape == (1000,) and (betas > 0).all() and betas.max() <= 0.999
    ac = np.cumprod(1 - betas)
    assert np.all(np.diff(ac) < 0) and ac[0] > 0.9999 and ac[-1] < 1e-4


def test_loss_and_inversion(real_plans):
    diffusion = Diffusion(1000)
    batch = PlanTensors.from_plans(real_plans).batch(torch.arange(7))
    t = torch.randint(0, 1000, (7,))
    noise = torch.randn_like(batch.x0)
    xt = diffusion.q_sample(batch.x0, t, noise)
    assert torch.allclose(diffusion.predict_x0(noise, xt, t), batch.x0, atol=1e-3)
    assert torch.allclose(diffusion.predict_eps(batch.x0, xt, t), noise, atol=1e-3)
    loss = diffusion.loss(tiny_model(), batch, torch.Generator().manual_seed(0))
    assert loss.item() == pytest.approx(1.0, abs=0.1)  # zero output vs unit noise


class Oracle(torch.nn.Module):
    """Returns the exact noise for known clean corners: a perfect denoiser."""

    def __init__(self, diffusion, x0):
        super().__init__()
        self.diffusion, self.x0 = diffusion, x0

    def forward(self, x, t, batch):
        return self.diffusion.predict_eps(self.x0, x, t)


@pytest.mark.parametrize(
    ("method", "steps", "eta"), [("ddpm", None, 0.0), ("ddim", 50, 0.0), ("ddim", 20, 1.0)]
)
def test_samplers_recover_the_data_with_a_perfect_denoiser(real_plans, method, steps, eta):
    diffusion = Diffusion(1000)
    batch = PlanTensors.from_plans(real_plans).batch(torch.arange(7))
    x0 = batch.x0 * batch.valid[..., None]
    out = diffusion.sample(
        Oracle(diffusion, x0), batch, method, steps, eta, torch.Generator().manual_seed(0)
    )
    assert torch.allclose(out, x0, atol=1e-4)


def test_generate_returns_well_formed_plans(real_plans):
    diffusion = Diffusion(50)
    graphs = [RoomGraph.from_plan(p) for p in real_plans[:3]]
    samples = generate(
        tiny_model(), diffusion, graphs, per_graph=2, method="ddim", steps=5, batch_size=4
    )
    assert [len(s) for s in samples] == [2, 2, 2]
    for graph, plans in zip(graphs, samples, strict=True):
        for k, plan in enumerate(plans):
            assert plan.meta["condition"] == graph.id and plan.meta["sample"] == k
            assert [r.type for r in plan.rooms] == list(graph.types)
            assert [len(r.polygon) for r in plan.rooms] == list(
                graph.corners
            )  # right corner counts
            coords = [v for r in plan.rooms for p in r.polygon for v in p]
            assert all(
                isinstance(v, int) and 0 <= v <= GRID_SIZE - 1 for v in coords
            )  # on the grid
    again = generate(
        tiny_model(), diffusion, graphs, per_graph=2, method="ddim", steps=5, batch_size=4
    )
    assert [p.rooms for s in again for p in s] == [p.rooms for s in samples for p in s]


# ---------------------------------------------------------------- EMA


def test_ema():
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = EMA(model, decay=0.5)
    model.weight.data.fill_(1.0)
    ema.update(model, step=1000)  # decay 0.5: halfway
    assert ema.model.weight.tolist() == [[0.5, 0.5]]
    ema.update(model, step=0)  # early steps: decay (1 + 0) / 10 = 0.1
    assert ema.model.weight[0, 0].item() == pytest.approx(0.95)
    assert not ema.model.weight.requires_grad
    copy = EMA(torch.nn.Linear(2, 1, bias=False), decay=0.9)
    copy.load_state_dict(ema.state_dict())
    assert copy.decay == 0.5 and torch.equal(copy.model.weight, ema.model.weight)


def test_generation_without_final_rounding(real_plans):
    graphs = [RoomGraph.from_plan(p) for p in real_plans[:2]]
    rounded = generate(tiny_model(), Diffusion(50), graphs, method="ddim", steps=5)
    floats = generate(tiny_model(), Diffusion(50), graphs, method="ddim", steps=5, discretize=False)
    for a, b in zip(rounded, floats, strict=True):
        for room_a, room_b in zip(a[0].rooms, b[0].rooms, strict=True):
            assert all(isinstance(v, float) for p in room_b.polygon for v in p)
            assert np.allclose(np.rint(np.asarray(room_b.polygon)), np.asarray(room_a.polygon))


def test_sampler_callback(real_plans):
    diffusion = Diffusion(100)
    batch = PlanTensors.from_plans(real_plans[:1]).batch(torch.arange(1))
    seen = []
    out = diffusion.sample(
        tiny_model(), batch, "ddim", 10, callback=lambda i, x, x0: seen.append(i)
    )
    assert seen == sorted(seen, reverse=True) and len(seen) == 10 and seen[-1] == 0
    assert out.shape == batch.x0.shape
    seen.clear()
    diffusion.sample(tiny_model(), batch, "ddpm", callback=lambda i, x, x0: seen.append(i))
    assert seen == list(range(99, -1, -1))
