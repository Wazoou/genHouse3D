"""Swiss Dwellings adapter on synthetic raw data (no dataset download needed)."""

from __future__ import annotations

import csv
import json

import pytest
from shapely.geometry import Polygon, box

from conftest import raw_apartment
from housegen.data import build
from housegen.data.schema import read_plans, validate_plan
from housegen.data.swiss_dwellings import (
    SwissDwellingsConfig,
    convert_apartment,
    iter_raw_apartments,
)

CFG = SwissDwellingsConfig()


@pytest.mark.parametrize("angle", [0.0, 30.0, -75.0])
def test_convert_apartment(angle):
    plan = convert_apartment(raw_apartment(angle), CFG)
    assert not isinstance(plan, str), plan
    assert validate_plan(plan) == []
    # The shaft is dropped, ROOM becomes a bedroom, rooms keep the raw order.
    assert [r.type for r in plan.rooms] == ["living", "kitchen", "bedroom", "bathroom"]
    assert all(len(r.polygon) == 4 for r in plan.rooms)
    # Walls are closed: the four rooms tile the 8 m x 8 m apartment (80 x 80 units).
    assert sum(plan.room_area_m2(i) for i in range(4)) == pytest.approx(64.0, abs=0.5)
    assert plan.edges() == [(0, 1, "open"), (0, 2, "door"), (0, 3, "door")]
    assert plan.entrance_rooms() == [0]
    assert len(plan.doors) == 3
    widths = sorted(round(d.line().length) for d in plan.doors)
    assert widths == [8, 9, 10]  # 0.8 m, 0.9 m and 1.0 m doors, in 10 cm units
    assert plan.id == "sd_1_abcdef012345" and plan.group == "sd_site_1"
    assert plan.meta["raw_types"] == ["LIVING_ROOM", "KITCHEN", "ROOM", "BATHROOM"]
    assert plan.meta["living_inferred"] is False


def test_plan_is_centred_on_the_grid():
    plan = convert_apartment(raw_apartment(0.0), CFG)
    xs = [x for r in plan.rooms for x, _ in r.polygon]
    ys = [y for r in plan.rooms for _, y in r.polygon]
    assert (min(xs), max(xs)) == (87, 167) and (min(ys), max(ys)) == (87, 167)


def test_largest_generic_room_becomes_the_living_room():
    raw = raw_apartment()
    raw.rooms = [("ROOM" if t == "LIVING_ROOM" else t, g) for t, g in raw.rooms]
    plan = convert_apartment(raw, CFG)
    assert [r.type for r in plan.rooms] == ["living", "kitchen", "bedroom", "bathroom"]
    assert plan.meta["living_inferred"] is True


def test_rejections():
    raw = raw_apartment()
    raw.rooms.append(("STAIRCASE", box(30, 30, 32, 32)))
    assert convert_apartment(raw, CFG) == "unsupported_room_type"

    raw = raw_apartment()
    raw.rooms[1] = ("KITCHEN", Polygon([(5, 0), (8, 0), (6.5, 4)]))
    assert convert_apartment(raw, CFG) == "non_rectilinear"

    # A 0.9 m wall is wider than max_wall_m: it is not closed, so the door crossing it
    # links two rooms that share no wall.
    raw = raw_apartment(angle_deg=0.0, wall=0.9)
    raw.doors[0] = (False, box(0.8, 4, 2.2, 4.9))
    assert convert_apartment(raw, CFG) == "door_without_shared_wall"

    raw = raw_apartment()
    raw.rooms = [(t, g) for t, g in raw.rooms if t == "SHAFT"]
    assert convert_apartment(raw, CFG) == "no_room"


COLUMNS = [
    "apartment_id",
    "site_id",
    "building_id",
    "plan_id",
    "floor_id",
    "unit_id",
    "area_id",
    "unit_usage",
    "entity_type",
    "entity_subtype",
    "geometry",
    "elevation",
    "height",
]


def _write_csv(path, floors=("3",)):
    """Raw CSV with the synthetic apartment on each floor of ``floors`` plus noise rows."""
    raw = raw_apartment()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        for floor in floors:
            base = {"site_id": "1", "building_id": "2", "floor_id": floor}
            for subtype, geom in raw.rooms:
                writer.writerow(
                    base
                    | {
                        "apartment_id": "abcdef0123456789",
                        "entity_type": "area",
                        "unit_usage": "RESIDENTIAL",
                        "entity_subtype": subtype,
                        "geometry": geom.wkt,
                    }
                )
            for entrance, geom in raw.doors:
                row = base | {
                    "entity_type": "opening",
                    "geometry": geom.wkt,
                    "entity_subtype": "ENTRANCE_DOOR" if entrance else "DOOR",
                }
                writer.writerow(row | {"unit_usage": "RESIDENTIAL"})
                writer.writerow(row | {"unit_usage": "PUBLIC"})  # listed once per unit
            writer.writerow(
                base
                | {
                    "entity_type": "separator",
                    "entity_subtype": "WALL",
                    "geometry": box(0, 4, 4, 4.15).wkt,
                }
            )
            writer.writerow(
                base
                | {
                    "entity_type": "area",
                    "unit_usage": "PUBLIC",
                    "entity_subtype": "STAIRCASE",
                    "geometry": box(9, 0, 12, 4).wkt,
                }
            )


def test_iter_raw_apartments(tmp_path):
    _write_csv(tmp_path / "geometries.csv")
    (raw,) = list(iter_raw_apartments(tmp_path / "geometries.csv"))
    assert (raw.site_id, raw.building_id, raw.floor_id) == ("1", "2", "3")
    assert [t for t, _ in raw.rooms] == ["LIVING_ROOM", "KITCHEN", "ROOM", "BATHROOM", "SHAFT"]
    assert len(raw.doors) == 3  # duplicates merged, the far-away door is not a candidate
    assert sum(entrance for entrance, _ in raw.doors) == 1
    assert not isinstance(convert_apartment(raw, CFG), str)


def test_apartments_on_several_floors_are_reported(tmp_path):
    _write_csv(tmp_path / "geometries.csv", floors=("3", "4"))
    assert list(iter_raw_apartments(tmp_path / "geometries.csv")) == ["multi_floor"]


def test_build_end_to_end(tmp_path):
    _write_csv(tmp_path / "geometries.csv")
    cfg = build.load_config("configs/data/swiss_dwellings.yaml")
    cfg |= {
        "raw_csv": str(tmp_path / "geometries.csv"),
        "processed_dir": str(tmp_path / "processed"),
        "splits_dir": str(tmp_path / "splits"),
    }
    stats = build.build(cfg)
    assert stats["raw_dwellings"] == stats["converted"] == stats["kept"] == 1
    assert stats["rejected"] == {} and stats["duplicates_removed"] == 0
    (plan,) = read_plans(build.plans_path(cfg))
    assert validate_plan(plan) == []
    assert sum(stats["splits"].values()) == 1
    split = next(name for name, count in stats["splits"].items() if count)
    assert build.load_split(cfg, split) == [plan]
    assert json.loads((tmp_path / "processed" / "build_stats.json").read_text()) == stats
