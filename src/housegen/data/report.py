"""``make viz``: data report (distributions and sample plans) for a built dataset.

Usage: ``python -m housegen.data.report --config configs/data/swiss_dwellings.yaml``
Writes ``reports/data_report.md`` and the figures under ``reports/figures/data/``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from housegen.data.build import load_config, plans_path
from housegen.data.schema import ROOM_TYPES, Plan, read_plans
from housegen.data.splits import SPLITS, read_split
from housegen.viz import ROOM_COLORS, plot_plans

REASONS = {
    "non_rectilinear": "a room has a slanted or curved wall (the format is rectilinear)",
    "too_many_rooms": "more rooms than `filters.max_rooms`",
    "too_few_rooms": "fewer rooms than `filters.min_rooms`",
    "multi_floor": "apartment spread over several floors (duplex)",
    "too_many_corners": "a room has more corners than `filters.max_corners_per_room`",
    "disconnected_graph": "a room cannot be reached through doors or openings",
    "unsupported_room_type": "contains an area outside the type table (staircase, office...)",
    "door_without_shared_wall": "a door links two rooms that share no wall after wall closing",
    "no_entrance": "no entrance door could be placed on an outer wall",
    "overlapping_rooms": "rooms overlap after wall closing",
    "degenerate_room": "a room collapses when snapped to the grid",
    "too_large": "does not fit in the 25.6 m canvas",
    "invalid_plan": "fails the validity check of the unified format",
    "invalid_raw_geometry": "raw geometry is not a simple polygon",
    "no_room": "no room left after dropping shafts and voids",
}


def _figure(width: float = 6.0, height: float = 3.4):
    fig = Figure(figsize=(width, height), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)


def _bar(counter: Counter, xlabel: str, path: Path, color: str = "#5b7fa6") -> None:
    fig, ax = _figure()
    keys = sorted(counter)
    ax.bar([str(k) for k in keys], [counter[k] for k in keys], color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    _save(fig, path)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def make_report(cfg: dict, out_md: Path, fig_dir: Path, n_examples: int = 20) -> None:
    plans: list[Plan] = read_plans(plans_path(cfg))
    stats = json.loads((Path(cfg["processed_dir"]) / "build_stats.json").read_text("utf-8"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel = fig_dir.relative_to(out_md.parent).as_posix()

    rooms_per_plan = Counter(len(p.rooms) for p in plans)
    type_counts = Counter(r.type for p in plans for r in p.rooms)
    plans_with = Counter(t for p in plans for t in {r.type for r in p.rooms})
    corners = Counter(len(r.polygon) for p in plans for r in p.rooms)
    total_corners = np.array([sum(len(r.polygon) for r in p.rooms) for p in plans])
    areas: dict[str, list[float]] = {t: [] for t in ROOM_TYPES}
    plan_area, extent = [], []
    edges = Counter()
    for p in plans:
        total = 0.0
        for i, room in enumerate(p.rooms):
            a = p.room_area_m2(i)
            areas[room.type].append(a)
            total += a if room.type != "balcony" else 0.0
        plan_area.append(total)
        pts = np.concatenate([np.asarray(r.polygon) for r in p.rooms])
        extent.append((pts.max(axis=0) - pts.min(axis=0)).max() * p.meters_per_unit)
        edges.update(kind for _, _, kind in p.edges())
    plan_area, extent = np.array(plan_area), np.array(extent)
    bedrooms = Counter(sum(r.type == "bedroom" for r in p.rooms) for p in plans)
    inferred = sum(bool(p.meta.get("living_inferred")) for p in plans)

    _bar(rooms_per_plan, "rooms per plan", fig_dir / "rooms_per_plan.png")
    _bar(corners, "corners per room", fig_dir / "corners_per_room.png")
    _bar(bedrooms, "bedrooms per plan", fig_dir / "bedrooms_per_plan.png")

    fig, ax = _figure()
    ax.bar(
        ROOM_TYPES,
        [type_counts[t] for t in ROOM_TYPES],
        color=[ROOM_COLORS[t] for t in ROOM_TYPES],
        edgecolor="#2b2b2b",
    )
    ax.set_ylabel("rooms")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, fig_dir / "room_types.png")

    fig, ax = _figure()
    box = ax.boxplot(
        [areas[t] for t in ROOM_TYPES],
        tick_labels=ROOM_TYPES,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#2b2b2b"},
    )
    for patch, t in zip(box["boxes"], ROOM_TYPES, strict=True):
        patch.set_facecolor(ROOM_COLORS[t])
    ax.set_ylabel("room area (m²)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, fig_dir / "area_by_type.png")

    fig, ax = _figure()
    ax.hist(plan_area, bins=40, color="#5b7fa6")
    ax.set_xlabel("indoor floor area per plan (m², balconies excluded)")
    ax.set_ylabel("plans")
    _save(fig, fig_dir / "plan_area.png")

    fig, ax = _figure()
    ax.hist(
        total_corners,
        bins=range(int(total_corners.min()), int(total_corners.max()) + 2),
        color="#5b7fa6",
    )
    ax.set_xlabel("corners per plan (sequence length seen by the model)")
    ax.set_ylabel("plans")
    _save(fig, fig_dir / "corners_per_plan.png")

    train_ids = set(read_split(cfg["splits_dir"], cfg["name"], "train"))
    train = [p for p in plans if p.id in train_ids]
    examples = random.Random(0).sample(train, n_examples)
    plot_plans(
        examples,
        fig_dir / "examples.png",
        ncols=5,
        titles=[f"{p.id}  ({len(p.rooms)} rooms)" for p in examples],
    )

    def pct(x: float) -> str:
        return f"{100 * x:.1f} %"

    def quantiles(values) -> list[str]:
        return [f"{v:.1f}" for v in np.percentile(values, [5, 25, 50, 75, 95])]

    src = cfg["source"]
    n = len(plans)
    md = [
        f"# Data report: {cfg['name']}",
        "",
        "Generated by `make viz` (`python -m housegen.data.report`). Do not edit by hand.",
        "",
        f"Source: [{src['doi']}]({src['url']}), licence {src['license']}. One plan is one "
        "single-storey apartment, rotated onto the axes and snapped to a 10 cm grid "
        "(256 x 256 units, 25.6 m). Rooms share edges: walls have no thickness in this format.",
        "",
        "## From raw dwellings to plans",
        "",
        _table(
            ["Step", "Dwellings"],
            [
                ["Residential apartments in the raw file", stats["raw_dwellings"]],
                ["Converted to the unified format", stats["converted"]],
                ["Passing the filters", stats["kept"] + stats["duplicates_removed"]],
                [
                    "Duplicates removed (same layout up to rotation / mirroring)",
                    stats["duplicates_removed"],
                ],
                ["**Kept**", f"**{stats['kept']}**"],
            ],
        ),
        "",
        "Rejections, most frequent first:",
        "",
        _table(
            ["Reason", "Dwellings", "Meaning"],
            [[f"`{k}`", v, REASONS.get(k, "")] for k, v in stats["rejected"].items()],
        ),
        "",
        "## Split",
        "",
        "Plans are split by site (a site groups the buildings of one development, which often "
        "share layouts), so near-identical plans never sit on both sides. Ids are frozen in "
        f"`{cfg['splits_dir']}/`.",
        "",
        _table(
            ["Split", "Plans", "Share", "Sites"],
            [
                [s, stats["splits"][s], pct(stats["splits"][s] / n), stats["groups"][s]]
                for s in SPLITS
            ],
        ),
        "",
        "## Rooms",
        "",
        f"![rooms per plan]({rel}/rooms_per_plan.png)",
        f"![room types]({rel}/room_types.png)",
        "",
        _table(
            ["Type", "Rooms", "Plans with at least one", "Median area (m²)"],
            [
                [
                    t,
                    type_counts[t],
                    pct(plans_with[t] / n),
                    f"{np.median(areas[t]):.1f}" if areas[t] else "-",
                ]
                for t in ROOM_TYPES
            ],
        ),
        "",
        f"The dataset labels many rooms with a generic `ROOM` type. In {inferred} plans "
        f"({pct(inferred / n)}) no living room is annotated and the largest generic room was "
        "relabelled `living`; the other generic rooms count as `bedroom`.",
        "",
        f"![bedrooms per plan]({rel}/bedrooms_per_plan.png)",
        "",
        "## Areas",
        "",
        f"![area by type]({rel}/area_by_type.png)",
        f"![plan area]({rel}/plan_area.png)",
        "",
        _table(
            ["Quantity", "5 %", "25 %", "median", "75 %", "95 %"],
            [
                ["Indoor floor area per plan (m²)", *quantiles(plan_area)],
                ["Largest side of the bounding box (m)", *quantiles(extent)],
            ],
        ),
        "",
        "## Corners",
        "",
        f"![corners per room]({rel}/corners_per_room.png)",
        f"![corners per plan]({rel}/corners_per_plan.png)",
        "",
        f"Corners per plan: median {int(np.median(total_corners))}, "
        f"95th percentile {int(np.percentile(total_corners, 95))}, "
        f"maximum {int(total_corners.max())}.",
        "",
        "## Room graph",
        "",
        _table(
            ["Connection", "Count", "Per plan"],
            [[k, edges[k], f"{edges[k] / n:.2f}"] for k in ("door", "open")],
        ),
        "",
        "`door`: a door on the shared wall. `open`: the two rooms touch without a wall "
        "(open kitchen, corridor opening onto the living room).",
        "",
        f"## {n_examples} random training plans",
        "",
        f"![examples]({rel}/examples.png)",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="reports/data_report.md")
    parser.add_argument("--figures", default="reports/figures/data")
    args = parser.parse_args()
    make_report(load_config(args.config), Path(args.out), Path(args.figures))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
