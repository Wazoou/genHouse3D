"""Gaussian diffusion on corner coordinates: training loss, DDPM and DDIM sampling.

Follows Ho et al. 2020 (DDPM), Nichol and Dhariwal 2021 (cosine schedule) and Song et al.
2021 (DDIM). Coordinates live in [-1, 1]; the predicted clean corners are clipped to that
range at every step, and rounding to the integer grid is left to the caller.
"""

from __future__ import annotations

import numpy as np
import torch

from housegen.model.encoding import Batch


def cosine_betas(steps: int, s: float = 0.008) -> np.ndarray:
    t = np.linspace(0, steps, steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(1 - f[1:] / f[:-1], 0.0, 0.999)


def linear_betas(steps: int) -> np.ndarray:
    scale = 1000 / steps
    return np.linspace(scale * 1e-4, scale * 0.02, steps)


class Diffusion:
    """Noise schedule and the operations that use it."""

    def __init__(self, steps: int = 1000, schedule: str = "cosine", prediction: str = "eps"):
        if prediction not in ("eps", "x0"):
            raise ValueError("prediction must be 'eps' or 'x0'")
        betas = {"cosine": cosine_betas, "linear": linear_betas}[schedule](steps)
        alphas_cumprod = np.cumprod(1.0 - betas)
        prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1 - prev) / (1 - alphas_cumprod)
        self.steps, self.prediction = steps, prediction
        self._tables = {
            "sqrt_ac": np.sqrt(alphas_cumprod),
            "sqrt_one_minus_ac": np.sqrt(1 - alphas_cumprod),
            "ac": alphas_cumprod,
            "coef_x0": betas * np.sqrt(prev) / (1 - alphas_cumprod),
            "coef_xt": (1 - prev) * np.sqrt(1 - betas) / (1 - alphas_cumprod),
            # log of the posterior variance, its first (zero) entry replaced by the second
            "log_var": np.log(np.append(posterior_variance[1], posterior_variance[1:])),
        }
        self._cache: dict[tuple[str, torch.device], torch.Tensor] = {}

    def _at(self, name: str, t: torch.Tensor) -> torch.Tensor:
        key = (name, t.device)
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(
                self._tables[name], dtype=torch.float32, device=t.device
            )
        return self._cache[key][t].view(-1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._at("sqrt_ac", t) * x0 + self._at("sqrt_one_minus_ac", t) * noise

    def loss(self, model, batch: Batch, generator: torch.Generator | None = None) -> torch.Tensor:
        """Mean squared error of the prediction over the valid corners (uniform timesteps)."""
        x0 = batch.x0
        t = torch.randint(0, self.steps, (x0.shape[0],), generator=generator, device=x0.device)
        noise = torch.randn(x0.shape, generator=generator, device=x0.device)
        out = model(self.q_sample(x0, t, noise), t, batch).float()
        target = noise if self.prediction == "eps" else x0
        valid = batch.valid.float()
        return ((out - target).pow(2).sum(-1) * valid).sum() / (2 * valid.sum())

    def predict_x0(self, out: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.prediction == "x0":
            return out
        return (xt - self._at("sqrt_one_minus_ac", t) * out) / self._at("sqrt_ac", t)

    def predict_eps(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (xt - self._at("sqrt_ac", t) * x0) / self._at("sqrt_one_minus_ac", t)

    @torch.no_grad()
    def sample(
        self,
        model,
        batch: Batch,
        method: str = "ddpm",
        steps: int | None = None,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        clip: bool = True,
    ) -> torch.Tensor:
        """Clean corners (B, L, 2) for the graphs of ``batch``; padding is zero.

        ``ddpm`` runs all the training steps with the posterior variance; ``ddim`` runs
        ``steps`` evenly spaced steps (deterministic when ``eta`` is 0).
        """
        device, b = batch.valid.device, batch.valid.shape[0]
        x = torch.randn(batch.x0.shape, generator=generator, device=device)

        def clean(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            x0 = self.predict_x0(model(x, t, batch).float(), x, t)
            return x0.clamp(-1.0, 1.0) if clip else x0

        if method == "ddpm":
            for i in range(self.steps - 1, -1, -1):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                x0 = clean(x, t)
                x = self._at("coef_x0", t) * x0 + self._at("coef_xt", t) * x
                if i > 0:
                    noise = torch.randn(x.shape, generator=generator, device=device)
                    x = x + torch.exp(0.5 * self._at("log_var", t)) * noise
        elif method == "ddim":
            count = min(steps or 100, self.steps)
            times = np.unique(np.linspace(0, self.steps - 1, count).round().astype(int))[::-1]
            ac = self._tables["ac"]
            for k, i in enumerate(times):
                t = torch.full((b,), int(i), device=device, dtype=torch.long)
                x0 = clean(x, t)
                eps = self.predict_eps(x0, x, t)
                a = float(ac[i])
                a_prev = float(ac[times[k + 1]]) if k + 1 < len(times) else 1.0
                sigma = eta * np.sqrt((1 - a_prev) / (1 - a) * (1 - a / a_prev))
                x = np.sqrt(a_prev) * x0 + np.sqrt(max(1 - a_prev - sigma**2, 0.0)) * eps
                if sigma > 0:
                    x = x + sigma * torch.randn(x.shape, generator=generator, device=device)
        else:
            raise ValueError("method must be 'ddpm' or 'ddim'")
        return x * batch.valid[..., None]
