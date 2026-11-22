"""Generate plans from room graphs with a trained model.

Usage: ``python -m housegen.sample --checkpoint runs/base/ema.pt --split test --out out.jsonl.gz``

Each output plan records in ``meta.condition`` the id of the graph it was generated for, so
``python -m housegen.evaluate --samples NAME=out.jsonl.gz`` can score it. Plans are written
as the model produced them: corners rounded to the grid and doors attached where the graph
asks for them, without any geometric clean-up.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from housegen.data.build import load_config, load_split
from housegen.data.schema import GRID_SIZE, Plan, Room, write_plans
from housegen.graphs import RoomGraph
from housegen.model.diffusion import Diffusion
from housegen.model.encoding import PlanTensors, to_grid
from housegen.model.transformer import CornerTransformer
from housegen.postprocess import attach_doors


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True
) -> tuple[CornerTransformer, Diffusion, dict]:
    """Model (EMA weights by default), diffusion and config from ``last.pt`` or ``ema.pt``."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = CornerTransformer(**cfg["model"]).to(device).eval()
    weights = state["ema"]["model"] if use_ema or "model" not in state else state["model"]
    model.load_state_dict(weights)
    return model, Diffusion(**cfg["diffusion"]), cfg


def decode_rooms(x: np.ndarray, graph: RoomGraph, discretize: bool = True) -> list[Room]:
    """Rooms from model corners (L, 2): back to the grid, clipped to the canvas, and rounded
    to integer grid units unless ``discretize`` is False (ablation of the final rounding)."""
    grid = np.clip(to_grid(x), 0, GRID_SIZE - 1)
    rooms, k = [], 0
    for room_type, n in zip(graph.types, graph.corners, strict=True):
        if discretize:
            corners = tuple((int(a), int(b)) for a, b in np.rint(grid[k : k + n]))
        else:
            corners = tuple((round(float(a), 3), round(float(b), 3)) for a, b in grid[k : k + n])
        rooms.append(Room(room_type, corners))
        k += n
    return rooms


def generate(
    model: CornerTransformer,
    diffusion: Diffusion,
    graphs: Sequence[RoomGraph],
    per_graph: int = 1,
    method: str = "ddim",
    steps: int | None = 100,
    seed: int = 0,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    random_room_ids: bool = True,
    eta: float = 0.0,
    discretize: bool = True,
) -> list[list[Plan]]:
    """``per_graph`` plans for each graph, in the order of ``graphs``; deterministic per seed."""
    device = torch.device(device)
    jobs = [(g, s) for g in graphs for s in range(per_graph)]
    out: dict[str, list[Plan]] = {g.id: [] for g in graphs}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        data = PlanTensors.from_graphs([g for g, _ in chunk], device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed * 1_000_003 + start)
        batch = data.batch(torch.arange(len(chunk)), generator, room_ids=random_room_ids)
        x = diffusion.sample(model, batch, method, steps, eta, generator).cpu().numpy()
        for k, (graph, s) in enumerate(chunk):
            rooms = decode_rooms(x[k, : int(data.lengths[k])], graph, discretize)
            used = diffusion.steps if method == "ddpm" else steps
            meta = {
                "condition": graph.id,
                "edges": [list(e) for e in graph.edges],  # the requested graph, for clean-up
                "entrances": list(graph.entrances),
                "sample": s,
                "method": method,
                "steps": used,
                "seed": seed,
            }
            out[graph.id].append(
                attach_doors(f"{graph.id}_s{s}", rooms, graph, source="generated", meta=meta)
            )
    return [out[g.id] for g in graphs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/swiss_dwellings.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="only the first N graphs")
    parser.add_argument("--per-graph", type=int, default=5)
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-discretize", action="store_true", help="keep float corners")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg = load_checkpoint(args.checkpoint, device)
    plans = load_split(load_config(args.data_config), args.split)[: args.limit]
    graphs = [RoomGraph.from_plan(p) for p in plans]
    samples = generate(
        model, diffusion, graphs, args.per_graph, args.method, args.steps, args.seed, device,
        random_room_ids=cfg["augment"]["room_ids"], discretize=not args.no_discretize,
    )  # fmt: skip
    count = write_plans(args.out, [p for group in samples for p in group])
    print(f"wrote {count} plans to {args.out}")


if __name__ == "__main__":
    main()
