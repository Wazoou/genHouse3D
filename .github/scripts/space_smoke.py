"""CI smoke test of the Gradio demo: assemble a Space with a tiny model, build the UI and
run one generation through the callback (Gradio cannot be loaded on every dev machine)."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch

from housegen.model.transformer import CornerTransformer
from housegen.space import assemble
from housegen.train import load_train_config

tmp = Path(tempfile.mkdtemp())
cfg = load_train_config("configs/model/overfit.yaml")
cfg["model"] |= {"dim": 32, "depth": 1, "heads": 4, "fourier": 1}
model = CornerTransformer(**cfg["model"])
torch.save({"ema": {"decay": 0.9, "model": model.state_dict()}, "config": cfg, "step": 1},
           tmp / "ema.pt")  # fmt: skip
space = assemble(tmp / "ema.pt", tmp / "space")
os.chdir(space)
sys.path.insert(0, str(space))
spec = importlib.util.spec_from_file_location("app", space / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)  # builds the Blocks, does not launch
plan, shown, files, text = app.generate(2, 1, "any", "hip", 30, 0, False)
assert Path(plan).exists(), text
print("demo UI built; one generation:", text)
