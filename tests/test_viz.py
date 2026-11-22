"""2D rendering."""

from __future__ import annotations

import numpy as np

from conftest import square_plan
from housegen.viz import ROOM_COLORS, plot_plans, render_plan_image


def _rgb(hex_color: str) -> tuple[int, ...]:
    return tuple(int(hex_color[k : k + 2], 16) for k in (1, 3, 5))


def test_render_plan_image_is_deterministic_and_coloured(real_plans):
    image = render_plan_image(real_plans[2], size=128)
    assert image.shape == (128, 128, 3) and image.dtype == np.uint8
    assert np.array_equal(image, render_plan_image(real_plans[2], size=128))
    colours = {tuple(c) for c in image.reshape(-1, 3)}
    assert (255, 255, 255) in colours  # background
    for room_type in {r.type for r in real_plans[2].rooms}:
        assert _rgb(ROOM_COLORS[room_type]) in colours


def test_render_uses_the_full_grid_frame():
    # The toy plan spans units 100..180 of 256: the room pixels must sit in that window.
    image = render_plan_image(square_plan(), size=256, doors=False)
    rows, cols = np.nonzero((image != 255).any(axis=2))
    assert 95 <= cols.min() <= 101 and 179 <= cols.max() <= 185
    assert 255 - 185 <= rows.min() <= 255 - 179 and 255 - 101 <= rows.max() <= 255 - 95


def test_plot_plans_writes_a_figure(tmp_path, real_plans):
    out = tmp_path / "figures" / "grid.png"
    plot_plans(real_plans, out, ncols=4, titles=[p.id for p in real_plans])
    assert out.stat().st_size > 10_000


def test_collapsed_rooms_are_skipped():
    import dataclasses

    from housegen.data.schema import Room

    plan = square_plan()
    rooms = (*plan.rooms[:3], Room("bathroom", ()))
    image = render_plan_image(dataclasses.replace(plan, rooms=rooms), size=64)
    assert image.shape == (64, 64, 3)
    empty = dataclasses.replace(plan, rooms=tuple(Room(r.type, ()) for r in plan.rooms), doors=())
    assert (render_plan_image(empty, size=32) == 255).all()
