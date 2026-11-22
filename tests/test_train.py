"""Training loop: checkpoints, exact resumption, time budget, logging (CPU, tiny model)."""

from __future__ import annotations

import copy
import csv

import pytest

torch = pytest.importorskip("torch")

from housegen.sample import load_checkpoint  # noqa: E402
from housegen.train import load_train_config, lr_factor, run  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    cfg = load_train_config("configs/model/overfit.yaml")
    cfg["model"] |= {"dim": 32, "depth": 2, "heads": 4, "fourier": 2}
    cfg["diffusion"]["steps"] = 100
    cfg["augment"] = {"d4": True, "room_ids": True}
    cfg["train"] |= {
        "batch_size": 3, "grad_accum": 2, "total_steps": 6, "warmup_steps": 2, "log_every": 1,
        "eval_every": 3, "checkpoint_every_steps": None, "checkpoint_minutes": 30,
        "ema_decay": 0.9,
    }  # fmt: skip
    return cfg


def _states_equal(a: dict, b: dict) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _quiet(*_args) -> None:
    pass


def test_interrupted_training_resumes_exactly(tmp_path, cfg, real_plans):
    straight = run(cfg, tmp_path / "straight", "cpu", train_plans=real_plans,
                   val_plans=real_plans[:2], log=_quiet)  # fmt: skip
    assert straight == straight | {"step": 6, "finished": True}

    stopped = copy.deepcopy(cfg)
    stopped["train"]["total_steps"] = 6
    stopped["train"]["checkpoint_every_steps"] = 3
    # First session: stopped by its time budget right after step 1, as a Kaggle session
    # reaching its limit would be (the budget check writes a checkpoint first).
    first = copy.deepcopy(stopped)
    first["train"]["max_minutes"] = 0.0
    messages = []
    summary = run(first, tmp_path / "resumed", "cpu", train_plans=real_plans,
                  val_plans=real_plans[:2], log=messages.append)  # fmt: skip
    assert summary["step"] == 1 and not summary["finished"]
    assert "time budget reached at step 1" in messages[-1]
    assert not (tmp_path / "resumed" / "ema.pt").exists()
    # Second session: the same command picks up the checkpoint and finishes.
    messages = []
    summary = run(stopped, tmp_path / "resumed", "cpu", train_plans=real_plans,
                  val_plans=real_plans[:2], log=messages.append)  # fmt: skip
    assert messages[0].startswith("resumed from step 1")
    assert summary["step"] == 6 and summary["finished"]

    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    for key in ("model", "ema", "optimizer", "scheduler", "scaler"):
        assert _states_equal(a[key], b[key]), key
    assert a["step"] == b["step"] == 6

    with open(tmp_path / "straight" / "log.csv", newline="") as f:
        straight_log = list(csv.DictReader(f))
    with open(tmp_path / "resumed" / "log.csv", newline="") as f:
        resumed_log = list(csv.DictReader(f))
    assert [r["step"] for r in resumed_log] == [str(s) for s in range(1, 7)]
    for x, y in zip(straight_log, resumed_log, strict=True):
        assert x["loss"] == y["loss"] and x["val_loss"] == y["val_loss"]
    assert [bool(r["val_loss"]) for r in resumed_log] == [False, False, True, False, False, True]


def test_checkpoint_contents_and_ema_export(tmp_path, cfg, real_plans):
    run(cfg, tmp_path, "cpu", train_plans=real_plans, val_plans=None, log=_quiet)
    state = torch.load(tmp_path / "last.pt", weights_only=False)
    expected = {"model", "ema", "optimizer", "scheduler", "scaler", "rng", "step", "minutes",
                "config", "git_commit", "torch"}  # fmt: skip
    assert expected <= set(state)
    assert {"python", "numpy", "torch"} <= set(state["rng"])
    assert state["config"]["train"]["total_steps"] == 6
    model, diffusion, config = load_checkpoint(tmp_path / "ema.pt")
    assert diffusion.steps == 100 and config == state["config"]
    assert all(torch.equal(p, state["ema"]["model"][k]) for k, p in model.state_dict().items())
    raw, _, _ = load_checkpoint(tmp_path / "last.pt", use_ema=False)
    assert all(torch.equal(p, state["model"][k]) for k, p in raw.state_dict().items())


def test_no_resume_starts_over(tmp_path, cfg, real_plans):
    run(cfg, tmp_path, "cpu", train_plans=real_plans, log=_quiet)
    messages = []
    run(cfg, tmp_path, "cpu", resume=False, train_plans=real_plans, log=messages.append)
    assert not any(m.startswith("resumed") for m in messages)
    with open(tmp_path / "log.csv", newline="") as f:
        assert [r["step"] for r in csv.DictReader(f)] == [str(s) for s in range(1, 7)]


def test_training_reduces_the_loss(tmp_path, cfg, real_plans):
    cfg["augment"] = {"d4": False, "room_ids": False}
    cfg["train"] |= {"total_steps": 150, "batch_size": 2, "grad_accum": 1, "lr": 3e-3,
                     "warmup_steps": 10, "log_every": 25, "eval_every": 0}  # fmt: skip
    run(cfg, tmp_path, "cpu", train_plans=real_plans[:2], log=_quiet)
    with open(tmp_path / "log.csv", newline="") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f)]
    assert losses[-1] < 0.6 * losses[0]


def test_lr_schedule():
    assert lr_factor(0, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100, 0.1) == pytest.approx(0.55)
    assert lr_factor(100, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_factor(500, 10, 100, 0.1) == pytest.approx(0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_resume_on_cuda(tmp_path, cfg, real_plans):
    """Checkpoints are loaded onto the GPU: the CPU random state must survive that."""
    first = copy.deepcopy(cfg)
    first["train"]["max_minutes"] = 0.0
    run(first, tmp_path, "cuda", train_plans=real_plans, log=_quiet)
    messages = []
    summary = run(cfg, tmp_path, "cuda", train_plans=real_plans, log=messages.append)
    assert messages[0].startswith("resumed from step 1") and summary["finished"]
