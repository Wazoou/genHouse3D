"""Exponential moving average of the weights: the copy used for generation."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    """Shadow copy of a model; ``decay`` ramps up as ``(1 + step) / (10 + step)`` early on."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for shadow, param in zip(self.model.parameters(), model.parameters(), strict=True):
            shadow.lerp_(param.detach(), 1.0 - decay)
        for shadow, buffer in zip(self.model.buffers(), model.buffers(), strict=True):
            shadow.copy_(buffer)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.model.load_state_dict(state["model"])
