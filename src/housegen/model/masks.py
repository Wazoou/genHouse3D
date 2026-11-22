"""Attention masks of the corner transformer (after HouseDiffusion, Shabani et al. 2023).

Three masks, ``True`` where a corner may attend to another one:

- ``room``: corners of the same room;
- ``global``: every corner of the plan;
- ``relational``: corners of the same room and of the rooms connected to it in the room
  graph (HouseDiffusion relates rooms and door components; here doors are not generated,
  so connected rooms attend to each other directly).

Padding tokens never take part, except that every token attends to itself, which keeps the
softmax of padding rows finite.
"""

from __future__ import annotations

import torch

ATTENTIONS: tuple[str, ...] = ("room", "global", "relational")


def attention_masks(
    room: torch.Tensor, valid: torch.Tensor, adjacency: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Boolean masks of shape (B, 1, L, L) from room indices (B, L), validity (B, L) and the
    room adjacency (B, R, R)."""
    batch, length = room.shape
    pair = valid[:, :, None] & valid[:, None, :]
    same = (room[:, :, None] == room[:, None, :]) & pair
    b = torch.arange(batch, device=room.device)[:, None, None]
    linked = adjacency[b, room[:, :, None], room[:, None, :]] & pair
    eye = torch.eye(length, dtype=torch.bool, device=room.device)
    return {
        "room": (same | eye)[:, None],
        "global": (pair | eye)[:, None],
        "relational": (same | linked | eye)[:, None],
    }
