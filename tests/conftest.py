"""Shared test helpers. Tests never touch the network or a GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely import affinity
from shapely.geometry import box

from housegen.data.schema import OUTSIDE, Door, Plan, Room, read_plans
from housegen.data.swiss_dwellings import RawApartment

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def real_plans() -> list[Plan]:
    return read_plans(FIXTURES / "plans.jsonl")


def square_plan(plan_id: str = "toy", group: str = "g0", shift: int = 0) -> Plan:
    """Four 4 m x 4 m rooms around a cross of walls, entrance on the living room."""
    a, b, c = 100 + shift, 140 + shift, 180 + shift
    rooms = (
        Room("living", ((a, a), (b, a), (b, b), (a, b))),
        Room("kitchen", ((b, a), (c, a), (c, b), (b, b))),
        Room("bedroom", ((a, b), (b, b), (b, c), (a, c))),
        Room("bathroom", ((b, b), (c, b), (c, c), (b, c))),
    )
    doors = (
        Door((0, 2), ((a + 10, b), (a + 19, b))),
        Door((2, 3), ((b, b + 10), (b, b + 19))),
        Door((0, OUTSIDE), ((a + 10, a), (a + 20, a))),
    )
    return Plan(plan_id, "toy", group, rooms, doors, open_edges=((0, 1),))


def raw_apartment(angle_deg: float = 30.0, wall: float = 0.15) -> RawApartment:
    """Synthetic raw apartment: 8 m x 8 m, four rooms separated by ``wall``-thick walls.

    Living room (bottom left) has doors to the bedroom (top left) and the bathroom (top
    right), opens directly onto the kitchen (bottom right) and carries the entrance door.
    The whole apartment is rotated by ``angle_deg`` like a building on its site.
    """
    w = wall
    rooms = [
        ("LIVING_ROOM", box(0, 0, 5, 4)),
        ("KITCHEN", box(5, 0, 8, 4)),  # touches the living room: open connection
        ("ROOM", box(0, 4 + w, 4, 8)),
        ("BATHROOM", box(4 + w, 4 + w, 8, 8)),
        ("SHAFT", box(7.6, 7.6, 7.9, 7.9)),  # ignored
    ]
    doors = [
        (False, box(1.0, 4, 1.9, 4 + w)),  # living - bedroom
        (False, box(4.2, 4, 5.0, 4 + w)),  # living - bathroom
        (True, box(2.0, -0.2, 3.0, 0)),  # entrance on the living room
        (False, box(20, 20, 21, 20.2)),  # door of another apartment on the floor
    ]

    def rot(geom):
        return affinity.rotate(geom, angle_deg, origin=(3, 2))

    return RawApartment(
        site_id="1",
        building_id="2",
        floor_id="3",
        apartment_id="abcdef0123456789",
        rooms=[(t, rot(g)) for t, g in rooms],
        doors=[(e, rot(g)) for e, g in doors],
    )
