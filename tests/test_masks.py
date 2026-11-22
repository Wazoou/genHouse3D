"""Attention masks: who may attend to whom."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from housegen.graphs import RoomGraph  # noqa: E402
from housegen.model.encoding import PlanTensors  # noqa: E402
from housegen.model.masks import attention_masks  # noqa: E402
from housegen.model.transformer import Attention, CornerTransformer  # noqa: E402

# Three rooms in a row: 0 - 1 - 2. Rooms 0 and 2 are not connected.
CHAIN = RoomGraph(
    "chain", ("living", "corridor", "bedroom"), (4, 6, 4), ((0, 1, "door"), (1, 2, "door")), (0,)
)
SHORT = RoomGraph("short", ("living", "kitchen"), (4, 4), ((0, 1, "open"),), (0,))


def _batch():
    return PlanTensors.from_graphs([CHAIN, SHORT]).batch(torch.arange(2))


def test_masks_follow_rooms_and_graph():
    batch = _batch()
    masks = attention_masks(batch.room, batch.valid, batch.adjacency)
    assert set(masks) == {"room", "global", "relational"}
    assert all(m.shape == (2, 1, 14, 14) and m.dtype == torch.bool for m in masks.values())
    room0, room1, room2 = range(0, 4), range(4, 10), range(10, 14)
    rel, same, glob = (masks[k][0, 0] for k in ("relational", "room", "global"))
    for i in room0:
        assert all(same[i, j] for j in room0) and not any(same[i, j] for j in room1)
        assert all(rel[i, j] for j in [*room0, *room1])
        assert not any(rel[i, j] for j in room2)  # room 2 is not connected to room 0
        assert all(glob[i, j] for j in range(14))
    for i in room1:  # the corridor hears both of its neighbours
        assert all(rel[i, j] for j in range(14))
    assert torch.equal(rel, rel.T) and torch.equal(same, same.T)


def test_padding_never_takes_part():
    batch = _batch()
    masks = attention_masks(batch.room, batch.valid, batch.adjacency)
    for mask in masks.values():
        m = mask[1, 0]  # the short plan: 8 valid tokens, 6 padding tokens
        assert not m[:8, 8:].any() and not m[8:, :8].any()
        assert torch.equal(m[8:, 8:], torch.eye(6, dtype=torch.bool))  # padding sees itself only


def test_a_corner_does_not_hear_an_unconnected_room():
    """Relational attention: moving room 2 changes nothing for the corners of room 0."""
    torch.manual_seed(0)
    batch = _batch()
    mask = attention_masks(batch.room, batch.valid, batch.adjacency)["relational"]
    attention = Attention(16, 4)
    x = torch.randn(2, 14, 16)
    moved = x.clone()
    moved[0, 10:14] += torch.randn(4, 16)  # room 2 of the chain
    before, after = attention(x, mask), attention(moved, mask)
    assert torch.allclose(before[0, :4], after[0, :4], atol=1e-6)  # room 0: unchanged
    assert not torch.allclose(before[0, 4:10], after[0, 4:10])  # room 1 (connected): changed


def test_model_with_relational_attention_only_respects_the_graph():
    torch.manual_seed(0)
    model = CornerTransformer(dim=32, depth=2, heads=4, fourier=2, attentions=["relational"])
    for p in model.parameters():  # leave the zero initialisation, or every output is 0
        torch.nn.init.normal_(p, std=0.2)
    batch = _batch()
    x, t = torch.rand(2, 14, 2) * 2 - 1, torch.tensor([500, 500])
    moved = x.clone()
    moved[0, 10:14] += 0.3
    before, after = model(x, t, batch), model(moved, t, batch)
    # Two layers: room 2 reaches room 0 through room 1 (2 -> 1 in the first layer, 1 -> 0
    # in the second). With a single layer it cannot.
    assert not torch.allclose(before[0, :4], after[0, :4])
    one_layer = CornerTransformer(dim=32, depth=1, heads=4, fourier=2, attentions=["relational"])
    one_layer.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("blocks.1.")}
    )
    before, after = one_layer(x, t, batch), one_layer(moved, t, batch)
    assert torch.allclose(before[0, :4], after[0, :4], atol=1e-6)


def test_global_attention_reaches_every_room():
    torch.manual_seed(0)
    model = CornerTransformer(dim=32, depth=1, heads=4, fourier=2, attentions=["global"])
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    batch = _batch()
    x, t = torch.rand(2, 14, 2) * 2 - 1, torch.tensor([10, 10])
    moved = x.clone()
    moved[0, 10:14] += 0.3
    assert not torch.allclose(model(x, t, batch)[0, :4], model(moved, t, batch)[0, :4])
