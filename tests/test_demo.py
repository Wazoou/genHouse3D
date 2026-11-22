"""Demo engine (no UI): graph selection, generation, export."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from housegen.demo import Demo, graphs_from_json, graphs_to_json  # noqa: E402
from housegen.evaluate import load_eval_config  # noqa: E402
from housegen.geometry3d import load_building_config  # noqa: E402
from housegen.graphs import RoomGraph  # noqa: E402
from housegen.model.diffusion import Diffusion  # noqa: E402
from housegen.model.transformer import CornerTransformer  # noqa: E402


class Copier(torch.nn.Module):
    """Stands in for a trained model: denoises towards a fixed real plan."""

    def __init__(self, diffusion, plan):
        super().__init__()
        from housegen.model.encoding import PlanTensors

        self.diffusion = diffusion
        self.x0 = PlanTensors.from_plans([plan]).batch(torch.arange(1)).x0

    def forward(self, x, t, batch):
        return self.diffusion.predict_eps(self.x0.expand_as(x), x, t)


def test_graph_json_round_trip(real_plans):
    graphs = [RoomGraph.from_plan(p) for p in real_plans]
    assert graphs_from_json(json.loads(json.dumps(graphs_to_json(graphs)))) == graphs


def test_demo_builds_a_house(tmp_path, real_plans):
    plan = real_plans[3]
    graph = RoomGraph.from_plan(plan)
    diffusion = Diffusion(100)
    demo = Demo(Copier(diffusion, plan), diffusion, {"augment": {"room_ids": False}}, [graph],
                load_building_config(), load_eval_config()["min_area_m2"])  # fmt: skip
    bedrooms, bathrooms = graph.types.count("bedroom"), graph.types.count("bathroom")
    result = demo.run(bedrooms, bathrooms, "any", "hip", 30.0, seed=1, out_dir=tmp_path, steps=10)
    assert result.valid, result.text
    assert result.plan_image.exists() and result.house_glb.read_bytes()[:4] == b"glTF"
    assert result.cutaway_glb.stat().st_size < result.house_glb.stat().st_size
    assert graph.id in result.text and "hip roof" in result.text
    assert "closest one" not in result.text


def test_demo_reports_failures(tmp_path, real_plans):
    graph = RoomGraph.from_plan(real_plans[0])
    torch.manual_seed(0)
    model = CornerTransformer(dim=32, depth=1, heads=4, fourier=1)  # untrained: outputs zeros
    demo = Demo(model, Diffusion(20), {"augment": {"room_ids": True}}, [graph],
                load_building_config(), load_eval_config()["min_area_m2"])  # fmt: skip
    result = demo.run(5, 3, "open", "flat", 30.0, seed=0, out_dir=tmp_path, steps=3, attempts=2)
    assert not result.valid and not result.house_glb.exists()
    assert "closest one" in result.text and "No valid plan after 2 samples" in result.text
    assert result.plan_image.exists()


def test_space_folder(tmp_path):
    """The assembled Space runs the demo engine from its own files (Gradio itself is not
    imported: it cannot be loaded on every machine)."""
    import ast
    import subprocess
    import sys

    from housegen.space import assemble
    from housegen.train import load_train_config

    cfg = load_train_config("configs/model/overfit.yaml")
    cfg["model"] |= {"dim": 32, "depth": 1, "heads": 4, "fourier": 1}
    model = CornerTransformer(**cfg["model"])
    checkpoint = tmp_path / "ema.pt"
    torch.save({"ema": {"decay": 0.9, "model": model.state_dict()}, "config": cfg, "step": 1},
               checkpoint)  # fmt: skip
    space = assemble(checkpoint, tmp_path / "space")
    names = {p.name for p in space.iterdir()}
    assert {"app.py", "README.md", "requirements.txt", "housegen", "model", "configs",
            "graphs_test.json"} <= names  # fmt: skip
    assert "sdk: gradio" in (space / "README.md").read_text(encoding="utf-8")
    ast.parse((space / "app.py").read_text(encoding="utf-8"))
    script = (
        "from pathlib import Path\n"
        "from housegen.demo import Demo\n"
        "d = Demo.from_files('model/ema.pt', 'graphs_test.json', 'configs/building.yaml',"
        " 'configs/eval.yaml')\n"
        "r = d.run(2, 1, 'any', 'hip', 30.0, 0, 'out', steps=2, attempts=1)\n"
        "print(len(d.graphs), r.plan_image.exists())\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=space, capture_output=True,
                          text=True, timeout=300)  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1309", "True"]
