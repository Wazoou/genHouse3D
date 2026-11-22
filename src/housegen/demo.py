"""Engine of the online demo, free of any UI code (the Gradio app in ``app/`` wraps it).

The user asks for room counts; a real test graph with those counts is drawn (graph
selection V1), the model generates a new plan for it with DDIM on the CPU, the plan is
cleaned and extruded, and the house is exported as GLB, with and without its roof.

``python -m housegen.demo --export-graphs app/graphs_test.json`` writes the test graphs the
demo draws from (graphs only, no geometry).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen.geometry3d.export import build_house, load_building_config
from housegen.graphs import RoomGraph, has_open_kitchen, select_graph
from housegen.postprocess import clean_plan, validity
from housegen.sample import generate, load_checkpoint
from housegen.viz import draw_plan, legend_handles


def graphs_to_json(graphs: list[RoomGraph]) -> list[dict]:
    return [
        {
            "id": g.id,
            "types": list(g.types),
            "corners": list(g.corners),
            "edges": [list(e) for e in g.edges],
            "entrances": list(g.entrances),
        }
        for g in graphs
    ]


def graphs_from_json(items: list[dict]) -> list[RoomGraph]:
    return [
        RoomGraph(
            d["id"],
            tuple(d["types"]),
            tuple(d["corners"]),
            tuple((int(i), int(j), str(k)) for i, j, k in d["edges"]),
            tuple(d["entrances"]),
        )
        for d in items
    ]


@dataclass
class DemoResult:
    plan_image: Path
    house_glb: Path
    cutaway_glb: Path
    text: str
    valid: bool


class Demo:
    def __init__(self, model, diffusion, cfg: dict, graphs: list[RoomGraph], building: dict,
                 min_area: dict, device: str = "cpu") -> None:  # fmt: skip
        self.model, self.diffusion, self.cfg = model, diffusion, cfg
        self.graphs, self.building, self.min_area, self.device = graphs, building, min_area, device

    @classmethod
    def from_files(
        cls, checkpoint: str | Path, graphs: str | Path, building: str | Path, eval_cfg: str | Path
    ) -> Demo:
        import yaml

        torch.set_num_threads(max(1, torch.get_num_threads()))
        model, diffusion, cfg = load_checkpoint(checkpoint, "cpu")
        items = json.loads(Path(graphs).read_text(encoding="utf-8"))
        min_area = yaml.safe_load(Path(eval_cfg).read_text(encoding="utf-8"))["min_area_m2"]
        return cls(model, diffusion, cfg, graphs_from_json(items),
                   load_building_config(building), min_area)  # fmt: skip

    def run(
        self, bedrooms: int, bathrooms: int, kitchen: str, roof: str, pitch: float, seed: int,
        out_dir: str | Path, steps: int = 100, attempts: int = 4,
    ) -> DemoResult:  # fmt: skip
        """Generate one house; up to ``attempts`` samples are drawn until one is valid."""
        start = time.perf_counter()
        open_kitchen = {"open": True, "closed": False}.get(kitchen)
        graph, exact = select_graph(self.graphs, bedrooms, bathrooms, open_kitchen, seed)
        for attempt in range(attempts):
            [[raw]] = generate(
                self.model, self.diffusion, [graph], 1, "ddim", steps, seed * 101 + attempt,
                self.device, random_room_ids=self.cfg["augment"]["room_ids"],
            )  # fmt: skip
            plan, _ = clean_plan(raw, graph)
            verdict = validity(plan, self.min_area)
            if verdict == "valid":
                break
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"house_{graph.id}_{seed}"
        image = out / f"{stem}.png"
        fig = Figure(figsize=(5.2, 5.6), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.02, 0.12, 0.96, 0.86))
        draw_plan(plan, ax)
        fig.legend(handles=legend_handles(), loc="lower center", ncols=3, fontsize=7, frameon=False)
        fig.savefig(image)
        house_glb, cutaway_glb = out / f"{stem}.glb", out / f"{stem}_cutaway.glb"
        lines = []
        if not exact:
            lines.append("No test graph has exactly these counts: the closest one was used.")
        counts = {t: graph.types.count(t) for t in set(graph.types)}
        lines.append(
            f"Room graph {graph.id}: {len(graph.types)} rooms ("
            + ", ".join(f"{n} {t}" for t, n in sorted(counts.items()))
            + f"), {'open' if has_open_kitchen(graph) else 'closed'} kitchen."
        )
        if verdict == "valid":
            house = build_house(plan, self.building, roof=roof, pitch_deg=pitch, seed=seed)
            house.export(house_glb)
            build_house(plan, self.building, roof="none", seed=seed).export(cutaway_glb)
            area = sum(
                plan.room_area_m2(i) for i, r in enumerate(plan.rooms) if r.type != "balcony"
            )
            lines.append(
                f"Indoor area {area:.0f} m², {house.info['roof_built']} roof, "
                f"{house.info['windows']} windows. Sample {attempt + 1} of {attempts}."
            )
        else:
            lines.append(f"No valid plan after {attempts} samples (last one: {verdict}).")
        lines.append(f"Generated in {time.perf_counter() - start:.1f} s on CPU.")
        return DemoResult(image, house_glb, cutaway_glb, " ".join(lines), verdict == "valid")


def main() -> None:
    from housegen.data.build import load_config, load_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-graphs", required=True, help="output JSON path")
    parser.add_argument("--data-config", default="configs/data/swiss_dwellings.yaml")
    args = parser.parse_args()
    graphs = [RoomGraph.from_plan(p) for p in load_split(load_config(args.data_config), "test")]
    Path(args.export_graphs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.export_graphs).write_text(json.dumps(graphs_to_json(graphs)) + "\n", "utf-8")
    print(f"wrote {len(graphs)} graphs to {args.export_graphs}")


if __name__ == "__main__":
    main()
