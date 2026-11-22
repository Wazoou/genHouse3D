"""Training script. The same command runs locally and on Kaggle, and resumes on its own.

Usage: ``python -m housegen.train --config configs/model/base.yaml [--out runs/base] [--amp]``

Relaunching the same command continues from ``<out>/last.pt``. A checkpoint (model, EMA,
optimiser, gradient scaler, scheduler, random states, step, config, git commit) is written
every ``train.checkpoint_minutes`` and at the end; ``--max-minutes`` stops cleanly before a
session limit. Batch ``m`` and its noise depend only on ``(seed, m)``, so an interrupted run
continues exactly as an uninterrupted one would.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from housegen.data.build import load_config, load_split
from housegen.data.schema import Plan
from housegen.model.diffusion import Diffusion
from housegen.model.ema import EMA
from housegen.model.encoding import BatchSchedule, PlanTensors
from housegen.model.transformer import CornerTransformer

LOG_FIELDS = (
    "step", "loss", "val_loss", "lr", "grad_norm", "loss_scale", "steps_per_sec", "minutes",
    "skipped",
)  # fmt: skip


def load_train_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_model(cfg: dict) -> CornerTransformer:
    return CornerTransformer(**cfg["model"])


def build_diffusion(cfg: dict) -> Diffusion:
    return Diffusion(**cfg["diffusion"])


def git_commit() -> str:
    """Current commit, suffixed ``-dirty`` with uncommitted changes; ``unknown`` without git."""
    try:
        here = Path(__file__).parent
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=here
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, cwd=here,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if head.returncode:
        return "unknown"
    return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def step_generator(seed: int, m: int, device: torch.device) -> torch.Generator:
    """Random generator of micro-batch ``m``: augmentation, timesteps and noise."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(np.random.SeedSequence([seed, m]).generate_state(1)[0]))
    return generator


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay down to ``min_ratio`` of the peak rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def select_plans(cfg: dict) -> tuple[list[Plan], list[Plan]]:
    """Training plans (optionally a seeded subset) and validation plans."""
    data, seed = cfg["data"], cfg["train"]["seed"]
    data_cfg = load_config(data["config"])
    train = load_split(data_cfg, data["split"])
    if data.get("subset"):
        train = sorted(random.Random(seed).sample(train, data["subset"]), key=lambda p: p.id)
    count = cfg["train"].get("eval_plans", 512)
    if data.get("val_split"):
        val = load_split(data_cfg, data["val_split"])
        val = sorted(random.Random(seed).sample(val, min(count, len(val))), key=lambda p: p.id)
    else:  # no validation split: score the training plans themselves (overfitting test)
        val = train[:count]
    return train, val


@torch.no_grad()
def validation_loss(model, diffusion: Diffusion, data: PlanTensors, seed: int) -> float:
    """Loss on fixed plans with fixed timesteps and noise, so values are comparable."""
    was_training = model.training
    model.eval()
    total = count = 0.0
    for start in range(0, len(data), 256):
        batch = data.batch(torch.arange(start, min(start + 256, len(data))))
        loss = diffusion.loss(model, batch, step_generator(seed + 1, start, data.device))
        n = float(batch.valid.sum())
        total, count = total + float(loss) * n, count + n
    model.train(was_training)
    return total / count


def _atomic_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    # The global generators are not used for training (see step_generator); their states
    # are saved anyway so that any other code depending on them resumes identically.
    state = {"python": random.getstate(), "numpy": np.random.get_state(),  # noqa: NPY002
             "torch": torch.get_rng_state()}  # fmt: skip
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    # Checkpoints are loaded with map_location=<device>: bring the states back to the CPU.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _trim_log(path: Path, step: int) -> None:
    """Drop log rows written after the checkpoint we resume from."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    cfg: dict,
    out_dir: str | Path,
    device: str | None = None,
    amp: bool | None = None,
    resume: bool = True,
    train_plans: list[Plan] | None = None,
    val_plans: list[Plan] | None = None,
    log=print,
) -> dict:
    """Train (or resume training) as configured; return a short summary."""
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(tc.get("amp", False) if amp is None else amp) and device.type == "cuda"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out_dir / "last.pt", out_dir / "log.csv"
    seed = tc["seed"]
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if train_plans is None:
        train_plans, val_plans = select_plans(cfg)
    data = PlanTensors.from_plans(train_plans, device)
    val = PlanTensors.from_plans(val_plans, device) if val_plans else None
    schedule = BatchSchedule(data.lengths.cpu().numpy(), tc["batch_size"], seed)
    model = build_model(cfg).to(device)
    ema = EMA(model, tc["ema_decay"])
    diffusion = build_diffusion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total, warmup = tc["total_steps"], tc["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, warmup, total, tc.get("min_lr_ratio", 0.1))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    commit = git_commit()

    step, minutes_before = 0, 0.0
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _set_rng_state(state["rng"])
        step, minutes_before = state["step"], state["minutes"]
        _trim_log(log_path, step)
        log(f"resumed from step {step} ({minutes_before:.1f} min already trained)")
    elif log_path.exists():
        log_path.unlink()
    params = sum(p.numel() for p in model.parameters())
    log(f"{params / 1e6:.2f} M parameters, {len(data)} training plans, device {device}, "
        f"amp {amp}, steps {step} -> {total}")  # fmt: skip

    start = time.time()

    def minutes() -> float:
        return minutes_before + (time.time() - start) / 60

    def save() -> None:
        _atomic_save(
            {
                "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": _rng_state(), "step": step,
                "minutes": minutes(), "config": cfg, "git_commit": commit,
                "torch": torch.__version__,
            },
            ckpt_path,
        )  # fmt: skip

    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS, lineterminator="\n")
    if new_log:
        writer.writeheader()
    accum, losses, skipped, bad_in_a_row = tc.get("grad_accum", 1), [], 0, 0
    grad_norm, last_ckpt, window = float("nan"), time.time(), (time.time(), step)
    every_steps = tc.get("checkpoint_every_steps")
    stopped_early = False
    model.train()
    try:
        while step < total:
            optimizer.zero_grad(set_to_none=True)
            loss_value, finite = 0.0, True
            for micro in range(accum):
                m = step * accum + micro
                generator = step_generator(seed, m, device)
                batch = data.batch(
                    torch.from_numpy(schedule(m)), generator,
                    d4=cfg["augment"]["d4"], room_ids=cfg["augment"]["room_ids"],
                )  # fmt: skip
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = diffusion.loss(model, batch, generator)
                if not torch.isfinite(loss):
                    finite = False
                    break
                scaler.scale(loss / accum).backward()
                loss_value += float(loss.detach()) / accum
            if finite:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                ema.update(model, step)
                losses.append(loss_value)
                bad_in_a_row = 0
            else:  # skip the update, keep the schedule: the next step sees new batches
                skipped += 1
                bad_in_a_row += 1
                if bad_in_a_row > 20:
                    raise FloatingPointError("loss not finite for 20 steps in a row")
            scheduler.step()
            step += 1

            val_loss = None
            if val is not None and tc.get("eval_every") and step % tc["eval_every"] == 0:
                val_loss = validation_loss(ema.model, diffusion, val, seed)
            if step % tc["log_every"] == 0 or val_loss is not None or step == total:
                now = time.time()
                writer.writerow(
                    {
                        "step": step,
                        "loss": f"{np.mean(losses):.6f}" if losses else "",
                        "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "loss_scale": scaler.get_scale() if amp else 1.0,
                        "steps_per_sec": f"{(step - window[1]) / max(now - window[0], 1e-9):.2f}",
                        "minutes": f"{minutes():.2f}",
                        "skipped": skipped,
                    }
                )
                log_file.flush()
                losses, window = [], (now, step)
            due = (every_steps and step % every_steps == 0) or (
                time.time() - last_ckpt >= 60 * tc.get("checkpoint_minutes", 30)
            )
            if due or step == total:
                save()
                last_ckpt = time.time()
            max_minutes = tc.get("max_minutes")
            if (
                max_minutes is not None
                and (time.time() - start) / 60 >= max_minutes
                and step < total
            ):
                save()
                stopped_early = True
                log(f"time budget reached at step {step}: checkpoint saved, relaunch to resume")
                break
    except KeyboardInterrupt:
        save()
        log(f"interrupted at step {step}: checkpoint saved")
        raise
    finally:
        log_file.close()
    if step >= total and not stopped_early:
        _atomic_save(
            {"ema": ema.state_dict(), "config": cfg, "step": step, "git_commit": commit},
            out_dir / "ema.pt",
        )
    return {"step": step, "finished": step >= total, "minutes": round(minutes(), 2),
            "skipped": skipped, "parameters": params}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/model/<name>.yaml")
    parser.add_argument("--out", default=None, help="run directory (default: runs/<name>)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only)")
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_train_config(args.config)
    for key in ("total_steps", "max_minutes", "checkpoint_every_steps"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)
    out = args.out or f"runs/{cfg['name']}"
    summary = run(cfg, out, args.device, True if args.amp else None, not args.no_resume)
    print(summary)


if __name__ == "__main__":
    main()
