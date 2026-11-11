"""3D pipeline: prisms, wall layout, openings, roofs, house assembly, GLB export, rendering."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon, box

from conftest import square_plan
from housegen.geometry3d import build_house, load_building_config
from housegen.geometry3d.mesh import body_problems, extrude, triangulate
from housegen.geometry3d.openings import place_doors, place_windows
from housegen.geometry3d.roof import (
    RoofError,
    flat_roof,
    gable_roof,
    hip_roof,
    is_nearly_rectangular,
)
from housegen.geometry3d.walls import build_layout
from housegen.render import render_house

SHAPES = {
    "rectangle": box(0, 0, 10, 6),
    "square": box(0, 0, 6, 6),
    "L": Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]),
    "U": Polygon([(0, 0), (12, 0), (12, 8), (8, 8), (8, 3), (4, 3), (4, 8), (0, 8)]),
    "T": Polygon([(0, 6), (12, 6), (12, 10), (8, 10), (8, 16), (4, 16), (4, 10), (0, 10)]),
    "Z": Polygon([(0, 0), (6, 0), (6, 3), (9, 3), (9, 9), (3, 9), (3, 6), (0, 6)]),
    # Two blocks joined at a corner by a small neck: two reflex corners meet head-on.
    "pinch": Polygon([(0, 0), (5, 0), (5, 4.9), (10, 4.9), (10, 10), (4.9, 10), (4.9, 5), (0, 5)]),
    "courtyard-like": Polygon([(0, 0), (9, 0), (9, 9), (0, 9), (0, 6), (6, 6), (6, 3), (0, 3)]),
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_building_config()


def _volume(body) -> float:
    return float(trimesh.Trimesh(*body, process=False).volume)


# ---------------------------------------------------------------- prisms


def test_extrude_with_a_hole_is_a_valid_solid():
    polygon = Polygon(box(0, 0, 4, 3).exterior, [box(1, 1, 2, 2).exterior])
    body = extrude(polygon, 0.5, 2.5)
    assert body_problems(body) == []
    assert _volume(body) == pytest.approx((12 - 1) * 2.0)
    assert body[0][:, 2].min() == 0.5 and body[0][:, 2].max() == 2.5


def test_extrude_survives_a_polygon_touching_itself_at_a_point():
    # Two squares sharing one corner: without splitting that vertex, the vertical edge
    # above it would belong to four faces.
    bowtie = box(0, 0, 1, 1).union(box(1, 1, 2, 2))
    bodies = [extrude(p, 0, 1) for p in getattr(bowtie, "geoms", [bowtie])]
    assert all(body_problems(b) == [] for b in bodies)
    pinched = Polygon(box(0, 0, 3, 3).exterior, [[(0, 1.5), (1, 1), (1.5, 1.5), (1, 2)]])
    pts, _ = triangulate(pinched)
    assert len(pts) > len(set(map(tuple, pts.round(6))))  # the touching vertex is duplicated
    assert body_problems(extrude(pinched, 0, 1)) == []


def test_body_problems_reports_defects():
    vertices, faces = extrude(box(0, 0, 1, 1), 0, 1)
    assert body_problems((vertices, faces[:-1])) == ["not watertight"]
    assert body_problems((vertices, faces[:, ::-1])) == ["normals point inwards"]


# ---------------------------------------------------------------- roofs


def _chebyshev_to_boundary(polygon: Polygon, x: float, y: float) -> float:
    """L-infinity distance from (x, y) to the boundary of a rectilinear polygon."""
    c = np.asarray(polygon.exterior.coords)
    best = math.inf
    for (x0, y0), (x1, y1) in pairwise(c):
        dx = max(min(x0, x1) - x, 0.0, x - max(x0, x1))
        dy = max(min(y0, y1) - y, 0.0, y - max(y0, y1))
        best = min(best, max(dx, dy))
    return best


def _height_at(body, x: float, y: float) -> float:
    """Highest point of the body above (x, y)."""
    vertices, faces = body
    best = -math.inf
    for a, b, c in vertices[faces]:
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
            best = max(best, u * a[2] + v * b[2] + (1 - u - v) * c[2])
    return best


@pytest.mark.parametrize("name", SHAPES)
def test_hip_roof_is_the_straight_skeleton_roof(name):
    """The roof is a valid solid whose height follows the straight skeleton.

    For a rectilinear polygon the straight-skeleton roof rises, at every point, by the
    L-infinity distance to the eaves times the slope: an independent closed form.
    """
    footprint, z, pitch, fascia = SHAPES[name], 2.5, 35.0, 0.15
    body = hip_roof(footprint, z, pitch, overhang=0.3, fascia=fascia, quantum=0.1)
    assert body_problems(body) == []
    eaves = footprint.buffer(0.3, join_style="mitre")
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = eaves.bounds
    checked = 0
    while checked < 150:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if not eaves.contains(Point(x, y)):
            continue
        expected = z + fascia + math.tan(math.radians(pitch)) * _chebyshev_to_boundary(eaves, x, y)
        assert _height_at(body, x, y) == pytest.approx(expected, abs=1e-6)
        checked += 1


def test_hip_roof_of_a_rectangle_has_four_faces_and_a_ridge():
    body = hip_roof(box(0, 0, 10, 6), 0.0, 45.0, overhang=0.0, fascia=0.1, quantum=0.1)
    vertices, _ = body
    assert vertices[:, 2].max() == pytest.approx(0.1 + 3.0)  # half the short side at 45 degrees
    ridge = vertices[np.isclose(vertices[:, 2], 3.1)]
    assert sorted(ridge[:, 0].round(6)) == [3.0, 7.0] and np.allclose(ridge[:, 1], 3.0)
    # Pyramid frustum formula: prism under the ridge plus two half pyramids.
    assert _volume(body) == pytest.approx(60 * 0.1 + (0.5 * 6 * 3 * 4 + 6 * 6 * 3 / 3))


def test_hip_roof_rejects_non_rectilinear_footprints():
    with pytest.raises(RoofError, match="rectilinear"):
        hip_roof(Polygon([(0, 0), (8, 0), (5, 6)]), 2.5)


def test_gable_roof_follows_the_long_side():
    for footprint, ridge_axis in ((box(0, 0, 10, 6), 0), (box(0, 0, 6, 10), 1)):
        roof, gable = gable_roof(footprint, 2.5, 30.0, overhang=0.3, thickness=0.15)
        assert body_problems(roof) == [] and body_problems(gable) == []
        rise = 3.0 * math.tan(math.radians(30.0))
        assert gable[0][:, 2].max() == pytest.approx(2.5 + rise)
        top = gable[0][np.isclose(gable[0][:, 2], 2.5 + rise)]
        assert np.ptp(top[:, ridge_axis]) == pytest.approx(10.0)  # the ridge spans the long side
        assert _volume(gable) == pytest.approx(0.5 * 6 * rise * 10)
        lo, hi = roof[0].min(axis=0), roof[0].max(axis=0)
        assert np.allclose(lo[:2], -0.3) and np.allclose(sorted(hi[:2]), [6.3, 10.3])


def test_flat_roof_and_rectangularity():
    slab, parapet = flat_roof(box(0, 0, 10, 6), 2.5, 0.25, 0.4, 0.2)
    assert body_problems(slab) == [] and body_problems(parapet) == []
    assert _volume(slab) == pytest.approx(60 * 0.25)
    assert _volume(parapet) == pytest.approx((60 - 9.6 * 5.6) * 0.4)
    assert is_nearly_rectangular(box(0, 0, 10, 6))
    assert not is_nearly_rectangular(SHAPES["L"])


# ---------------------------------------------------------------- layout and openings


def test_wall_layout_of_the_toy_plan(cfg):
    layout = build_layout(square_plan(), cfg)
    # 8 m x 8 m of rooms, 20 cm outer walls built outwards: 84 x 84 units.
    assert layout.outer.bounds == (-42.0, -42.0, 42.0, 42.0)
    assert layout.interior.area == pytest.approx(6400.0)
    # Outer band + three partitions (living / kitchen are open onto each other).
    assert layout.walls.area == pytest.approx((84**2 - 80**2) + 3 * 40 * 1.0 - 0.75, abs=1.0)
    assert not layout.walls.contains(Point(0, -20))  # no wall between living room and kitchen
    assert layout.walls.contains(Point(0, 20)) and layout.walls.contains(Point(-20, 0))


def test_doors_and_windows(cfg):
    layout = build_layout(square_plan(), cfg)
    doors = place_doors(layout, cfg)
    assert sorted(d.kind for d in doors) == ["door", "door", "entrance"]
    entrance = next(d for d in doors if d.kind == "entrance")
    assert entrance.width == pytest.approx(10.0)  # 1.0 m
    assert entrance.outward.tolist() == [0.0, -1.0]  # the entrance is on the south wall
    assert all(
        d.outward is None and d.width == pytest.approx(9.0) for d in doors if d.kind == "door"
    )

    windows = place_windows(layout, cfg, doors, np.random.default_rng(0))
    again = place_windows(layout, cfg, doors, np.random.default_rng(0))
    assert [(w.room, w.p0.tolist(), w.p1.tolist()) for w in windows] == [
        (w.room, w.p0.tolist(), w.p1.tolist()) for w in again
    ]
    facade = layout.interior.exterior
    margin = cfg["windows"]["margin"] / layout.unit
    for w in windows:
        assert layout.types[w.room] in cfg["windows"]["rules"]
        assert facade.distance(Point(w.p0)) < 1e-9 and facade.distance(Point(w.p1)) < 1e-9
        assert (
            w.z1 <= cfg["walls"]["height"] - 0.15
            and w.z0 == cfg["windows"]["rules"][layout.types[w.room]]["sill"]
        )
        # Clear of the entrance door and of the other windows.
        for other in [entrance, *(o for o in windows if o is not w)]:
            if abs(float((other.p1 - other.p0) @ (w.p1 - w.p0))) > 1e-9:  # same wall direction
                gap = Point((w.p0 + w.p1) / 2).distance(Point((other.p0 + other.p1) / 2))
                assert gap >= (w.width + other.width) / 2 + margin - 1e-6 or gap > 30
    rooms_with_windows = {layout.types[w.room] for w in windows}
    assert rooms_with_windows == {"living", "kitchen", "bedroom", "bathroom"}


# ---------------------------------------------------------------- houses


@pytest.mark.parametrize("roof", ["hip", "gable", "flat", "none"])
def test_real_plans_become_valid_houses(cfg, real_plans, roof):
    for plan in real_plans:
        house = build_house(plan, cfg, roof=roof, seed=0)
        assert house.problems() == [], plan.id
        assert house.info["roof_requested"] == roof
        if roof in ("hip", "flat", "none"):
            assert house.info["roof_built"] == roof and house.info["roof_error"] == ""
        else:
            assert house.info["roof_built"] in ("gable", "hip")
        assert (house.part("roof") is None) == (roof == "none")
        walls = house.part("walls").mesh()
        assert walls.bounds[0][2] == 0.0 and walls.bounds[1][2] == cfg["walls"]["height"]
        assert house.info["doors"] >= 1 and house.info["windows"] >= 1
        assert house.info["triangles"] == house.triangles() < 20_000


def test_openings_really_cut_the_walls(cfg):
    plan = square_plan()
    with_openings = build_house(plan, cfg, roof="none")
    solid_cfg = cfg | {"windows": cfg["windows"] | {"rules": {}}}
    no_windows = build_house(plan, solid_cfg, roof="none")
    assert with_openings.info["windows"] > 0 and no_windows.info["windows"] == 0
    assert with_openings.part("glass") is not None and no_windows.part("glass") is None
    assert with_openings.part("walls").mesh().volume < no_windows.part("walls").mesh().volume
    # The entrance door (1.0 x 2.1 m through a 0.2 m wall) is cut out in both.
    layout = build_layout(plan, cfg)
    full = layout.walls.area * layout.unit**2 * cfg["walls"]["height"]
    doors = 1.0 * 0.2 * 2.1 + 2 * 0.9 * 0.1 * 2.1
    assert no_windows.part("walls").mesh().volume == pytest.approx(full - doors, rel=1e-6)


def test_build_house_is_deterministic_and_checks_arguments(cfg, real_plans):
    a = build_house(real_plans[2], cfg, seed=3)
    b = build_house(real_plans[2], cfg, seed=3)
    for pa, pb in zip(a.parts, b.parts, strict=True):
        assert pa.name == pb.name
        assert all(np.array_equal(x[0], y[0]) for x, y in zip(pa.bodies, pb.bodies, strict=True))
    with pytest.raises(ValueError, match="roof type"):
        build_house(real_plans[0], cfg, roof="dome")
    with pytest.raises(ValueError, match="pitch"):
        build_house(real_plans[0], cfg, pitch_deg=60.0)


def test_steeper_pitch_gives_a_taller_roof(cfg, real_plans):
    low = build_house(real_plans[1], cfg, roof="hip", pitch_deg=25.0).part("roof").mesh()
    high = build_house(real_plans[1], cfg, roof="hip", pitch_deg=45.0).part("roof").mesh()
    assert high.bounds[1][2] > low.bounds[1][2] > cfg["walls"]["height"]


def test_glb_export_round_trip(cfg, real_plans, tmp_path):
    house = build_house(real_plans[3], cfg, roof="hip")
    path = house.export(tmp_path / "house.glb")
    assert path.read_bytes()[:4] == b"glTF"
    scene = trimesh.load(path)
    assert set(scene.geometry) == {p.name for p in house.parts}
    assert sum(len(g.faces) for g in scene.geometry.values()) == house.triangles()
    # glTF is y-up: the house height is along y, the floor slab starts below 0.
    lo, hi = scene.bounds
    assert lo[1] == pytest.approx(-cfg["floor"]["slab_thickness"])
    assert hi[1] == pytest.approx(house.part("roof").mesh().bounds[1][2])
    glass = scene.geometry["glass"].visual.material
    assert glass.alphaMode == "BLEND" and glass.baseColorFactor[3] < 255


def test_render_house(cfg, real_plans):
    house = build_house(real_plans[0], cfg, roof="hip")
    image = render_house(house, "iso", size=(160, 120), supersample=2)
    assert image.shape == (120, 160, 3) and image.dtype == np.uint8
    assert np.array_equal(image, render_house(house, "iso", size=(160, 120), supersample=2))
    roof_rgb = np.array(cfg["materials"]["roof"][:3]) * 255
    close_to_roof = np.abs(image.astype(float) - roof_rgb).sum(axis=2) < 120
    assert close_to_roof.mean() > 0.05  # the roof is visible
    top = render_house(build_house(real_plans[0], cfg, roof="none"), "top", size=(160, 120))
    assert top.shape == (120, 160, 3) and not np.array_equal(top, image)
