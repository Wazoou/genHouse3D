"""Geometry helpers: rotation, rectification, wall closing, ring cleaning."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

from housegen.data.geometry import clean_ring, dominant_angle, rectify_ring, snap_walls


def _ring(geom) -> np.ndarray:
    ring = rectify_ring(geom)
    assert ring is not None
    return ring


@pytest.mark.parametrize("angle", [-40.0, -10.0, 0.0, 20.0, 44.0])
def test_dominant_angle_recovers_the_rotation(angle):
    rooms = [affinity.rotate(g, angle, origin=(0, 0)) for g in (box(0, 0, 5, 3), box(5, 0, 8, 6))]
    assert math.degrees(dominant_angle(rooms)) == pytest.approx(angle, abs=1e-6)


def test_rectify_makes_a_noisy_room_rectilinear():
    noisy = Polygon([(0, 0), (2, 0.02), (4, -0.01), (4.02, 3), (2, 3.03), (0.01, 3)])
    ring = _ring(noisy)
    assert len(ring) == 4
    assert Polygon(ring).area == pytest.approx(12.0, rel=0.02)
    xs, ys = sorted(set(ring[:, 0].round(6))), sorted(set(ring[:, 1].round(6)))
    assert len(xs) == 2 and len(ys) == 2  # exactly axis aligned


def test_rectify_keeps_an_l_shape_and_orientation():
    l_shape = Polygon([(0, 0), (0, 4), (2, 4), (2, 2), (5, 2), (5, 0)])  # clockwise input
    ring = _ring(l_shape)
    assert len(ring) == 6
    assert Polygon(ring).exterior.is_ccw
    assert Polygon(ring).equals(l_shape)


def test_rectify_rejects_slanted_rooms():
    assert rectify_ring(Polygon([(0, 0), (4, 0), (4, 3), (2, 5), (0, 3)])) is None
    assert rectify_ring(affinity.rotate(box(0, 0, 4, 3), 30)) is None


def test_rectify_absorbs_a_small_chamfer():
    chamfered = Polygon([(0, 0), (4, 0), (4, 2.9), (3.9, 3), (0, 3)])
    assert len(_ring(chamfered)) == 4


def test_snap_walls_puts_neighbours_on_the_wall_centre_line():
    left, right = _ring(box(0, 0, 4, 3)), _ring(box(4.2, 0, 8, 3))
    a, b = snap_walls([left, right])
    assert a[:, 0].max() == pytest.approx(4.1) and b[:, 0].min() == pytest.approx(4.1)
    assert a[:, 0].min() == 0 and b[:, 0].max() == 8  # outer faces do not move
    assert Polygon(a).intersection(Polygon(b)).area == pytest.approx(0)
    assert Polygon(a).exterior.intersection(Polygon(b).exterior).length == pytest.approx(3)


def test_snap_walls_handles_a_wall_crossing():
    rooms = [box(0, 0, 4, 3), box(4.2, 0, 8, 3), box(0, 3.1, 5, 6), box(5.3, 3.1, 8, 6)]
    snapped = snap_walls([_ring(r) for r in rooms])
    union_area = sum(Polygon(r).area for r in snapped)
    assert union_area == pytest.approx(48.0)  # the four rooms tile the 8 x 6 rectangle
    for i in range(4):
        for j in range(i + 1, 4):
            assert Polygon(snapped[i]).intersection(Polygon(snapped[j])).area == pytest.approx(0)


def test_snap_walls_leaves_distant_or_separated_rooms_alone():
    far = snap_walls([_ring(box(0, 0, 4, 3)), _ring(box(5, 0, 8, 3))], max_wall=0.7)
    assert far[0][:, 0].max() == 4 and far[1][:, 0].min() == 5
    # A thin room sits between the two faces: they are not a wall.
    rings = [_ring(box(0, 0, 4, 3)), _ring(box(4.65, 0, 8, 3)), _ring(box(4.1, 0, 4.55, 3))]
    a, b, mid = snap_walls(rings, max_wall=0.7, align_tol=0.0)
    assert a[:, 0].max() == pytest.approx(4.05) and mid[:, 0].min() == pytest.approx(4.05)
    assert b[:, 0].min() == pytest.approx(4.6) and mid[:, 0].max() == pytest.approx(4.6)


def test_snap_walls_aligns_almost_collinear_faces():
    rings = [_ring(box(0, 0, 4, 3)), _ring(box(4, 0.04, 8, 3))]
    a, b = snap_walls(rings, align_tol=0.1)
    assert a[:, 1].min() == pytest.approx(b[:, 1].min())


def test_clean_ring_removes_spikes_and_fixes_orientation():
    spike = np.array([(0, 0), (4, 0), (4, 2), (6, 2), (4, 2), (4, 3), (2, 3), (0, 3)])
    assert clean_ring(spike).tolist() == [[0, 0], [4, 0], [4, 3], [0, 3]]
    clockwise = np.array([(0, 0), (0, 3), (4, 3), (4, 0)])
    assert Polygon(clean_ring(clockwise)).exterior.is_ccw
    assert clean_ring(np.array([(0, 0), (4, 0), (4, 0), (0, 0)])) is None
