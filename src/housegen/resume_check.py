"""Resume check (brief, section 12): kill a training run for real, relaunch it, compare.

Usage: ``python -m housegen.resume_check --config configs/model/overfit.yaml``

1. Starts ``python -m housegen.train`` in a subprocess with a checkpoint every
   ``--every`` steps, waits until the run is past its first checkpoint and hard-kills it
   (as a Kaggle session that ends: no clean shutdown).
2. Relaunches the very same command, which must resume from the checkpoint and finish.
3. Runs the same training without interruption and compares the two runs.

Writes ``reports/results/resume_check.csv``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def _last_logged_step(log: Path) -> int:
    if not log.exists():
        return 0
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    return int(lines[-1].split(",")[0]) if len(lines) > 1 else 0


def _command(config: str, out: Path, total: int, every: int) -> list[str]:
    return [sys.executable, "-m", "housegen.train", "--config", config, "--out", str(out),
            "--total-steps", str(total), "--checkpoint-every-steps", str(every)]  # fmt: skip


def _final_loss(log: Path) -> float:
    with open(log, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["loss"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/overfit.yaml")
    parser.add_argument("--out", default="runs/resume_check")
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--kill-after", type=int, default=700, help="kill once this step is logged")
    parser.add_argument("--results", default="reports/results")
    args = parser.parse_args()
    out = Path(args.out)
    killed, straight = out / "killed", out / "straight"
    shutil.rmtree(out, ignore_errors=True)

    process = subprocess.Popen(_command(args.config, killed, args.total, args.every))
    while _last_logged_step(killed / "log.csv") < args.kill_after:
        if process.poll() is not None:
            raise RuntimeError("training ended before it could be killed")
        time.sleep(0.5)
    process.kill()  # TerminateProcess / SIGKILL: no checkpoint on the way out
    process.wait()
    killed_at = _last_logged_step(killed / "log.csv")
    checkpoint_step = torch.load(killed / "last.pt", weights_only=False)["step"]
    print(f"killed at step {killed_at}, last checkpoint at step {checkpoint_step}")

    relaunch = subprocess.run(
        _command(args.config, killed, args.total, args.every), capture_output=True, text=True
    )
    print(relaunch.stdout)
    if relaunch.returncode:
        raise RuntimeError(relaunch.stderr)
    resumed_line = next(line for line in relaunch.stdout.splitlines() if "resumed" in line)
    subprocess.run(_command(args.config, straight, args.total, args.every), check=True,
                   capture_output=True)  # fmt: skip

    a = torch.load(killed / "last.pt", weights_only=False)
    b = torch.load(straight / "last.pt", weights_only=False)
    ema_a, ema_b = a["ema"]["model"], b["ema"]["model"]
    diffs = [(ema_a[k] - ema_b[k]).abs().max().item() for k in ema_a]
    with open(killed / "log.csv", newline="", encoding="utf-8") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    row = {
        "killed_at_logged_step": killed_at,
        "checkpoint_step": checkpoint_step,
        "relaunch_message": resumed_line.strip(),
        "final_step": a["step"],
        "log_steps_continuous": int(steps == sorted(set(steps))),
        "final_loss_interrupted": round(_final_loss(killed / "log.csv"), 6),
        "final_loss_uninterrupted": round(_final_loss(straight / "log.csv"), 6),
        "max_abs_ema_weight_difference": f"{max(diffs):.3e}",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(row)
    path = Path(args.results) / "resume_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
