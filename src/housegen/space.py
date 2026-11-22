"""Assemble the Hugging Face Space folder of the demo (``make space``).

Usage: ``python -m housegen.space --checkpoint runs/base/ema.pt --out outputs/space``

The folder holds everything the Space needs: ``app.py``, ``README.md`` (Space metadata),
``requirements.txt``, the ``housegen`` package, the EMA weights (fp32, the training state is
dropped), the test graphs and two config files. Upload it to a Gradio Space as it is.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

GRADIO_VERSION = "6.32.0"

README = f"""---
title: genHouse3D
emoji: 🏠
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

Novel 3D houses from room graphs: a floor-plan diffusion model followed by procedural 3D
extrusion. Code, evaluation and licences: https://github.com/Wazoou/genHouse3D

Floor plans are derived from Swiss Dwellings v3.0.0 (Archilyse AG,
https://doi.org/10.5281/zenodo.7788422), licensed under CC BY 4.0.
"""

REQUIREMENTS = f"""--extra-index-url https://download.pytorch.org/whl/cpu
torch
gradio=={GRADIO_VERSION}
matplotlib
networkx
numpy
pyyaml
shapely>=2.0.6
trimesh>=4
"""


def assemble(checkpoint: str | Path, out: str | Path, root: str | Path = ".") -> Path:
    root, out = Path(root), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy(root / "app" / "app.py", out / "app.py")
    shutil.copy(root / "app" / "graphs_test.json", out / "graphs_test.json")
    for name in ("building.yaml", "eval.yaml"):
        shutil.copy(root / "configs" / name, out / "configs" / name)
    shutil.copytree(
        root / "src" / "housegen", out / "housegen", ignore=shutil.ignore_patterns("__pycache__")
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {k: v.float() for k, v in state["ema"]["model"].items()}
    torch.save(
        {"ema": {"decay": state["ema"]["decay"], "model": weights}, "config": state["config"],
         "step": state["step"], "git_commit": state.get("git_commit", "unknown")},
        out / "model" / "ema.pt",
    )  # fmt: skip
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/base/ema.pt")
    parser.add_argument("--out", default="outputs/space")
    args = parser.parse_args()
    out = assemble(args.checkpoint, args.out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"assembled {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
