"""Assemble a 3D house from a plan and export it as GLB."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import trimesh
import yaml
from shapely import affinity
from shapely.ops import unary_union
from trimesh.visual.material import PBRMaterial

from housegen.data.geometry import straight_pieces
from housegen.data.schema import Plan
from housegen.geometry3d.mesh import Body, Part, body_problems, extrude_all, polygons
from housegen.geometry3d.openings import place_doors, place_windows
from housegen.geometry3d.roof import (
    RoofError,
    flat_roof,
    gable_roof,
    hip_roof,
    is_nearly_rectangular,
)
from housegen.geometry3d.walls import build_layout

ROOF_TYPES = ("hip", "gable", "flat", "none")
# glTF is y-up: (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def load_building_config(path: str | Path = "configs/building.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass
class House:
    """A 3D house: coloured parts in metres, z up, floor at z = 0."""

    plan_id: str
    parts: list[Part]
    info: dict = field(default_factory=dict)

    def part(self, name: str) -> Part | None:
        return next((p for p in self.parts if p.name == name), None)

    def problems(self) -> list[str]:
        """Mesh defects of every body (empty when the house is valid)."""
        return [
            f"{part.name}[{k}]: {problem}"
            for part in self.parts
            for k, body in enumerate(part.bodies)
            for problem in body_problems(body)
        ]

    def triangles(self) -> int:
        return sum(len(f) for part in self.parts for _, f in part.bodies)

    def scene(self) -> trimesh.Scene:
        scene = trimesh.Scene()
        for part in self.parts:
            mesh = part.mesh()
            mesh.visual = trimesh.visual.TextureVisuals(
                material=PBRMaterial(
                    name=part.name,
                    baseColorFactor=[round(255 * c) for c in part.color],
                    metallicFactor=0.0,
                    roughnessFactor=0.9,
                    alphaMode="BLEND" if part.color[3] < 1.0 else "OPAQUE",
                )
            )
            scene.add_geometry(mesh, node_name=part.name, geom_name=part.name)
        return scene

    def export(self, path: str | Path) -> Path:
        """Write the house as a binary glTF file (y up, as the format requires)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = self.scene()
        scene.apply_transform(Z_UP_TO_Y_UP)
        path.write_bytes(scene.export(file_type="glb"))
        return path


def build_house(
    plan: Plan,
    cfg: dict,
    roof: str | None = None,
    pitch_deg: float | None = None,
    seed: int = 0,
) -> House:
    """Extrude ``plan`` into a house: walls, doors, windows, floor, balconies and roof.

    ``roof`` (``hip``, ``gable``, ``flat`` or ``none`` for a cutaway view) and ``pitch_deg``
    default to the config. Window placement is pseudo-random and fixed by ``seed``. A roof
    that cannot be built falls back to the next simpler one (gable -> hip -> flat); what was
    built is reported in ``House.info``.
    """
    roof = roof or cfg["roof"]["type"]
    if roof not in ROOF_TYPES:
        raise ValueError(f"unknown roof type {roof!r}")
    pitch = cfg["roof"]["pitch_deg"] if pitch_deg is None else pitch_deg
    if not 25.0 <= pitch <= 45.0:
        raise ValueError("roof pitch must be between 25 and 45 degrees")
    layout = build_layout(plan, cfg)
    unit, height = layout.unit, cfg["walls"]["height"]
    colors = cfg["materials"]
    doors = place_doors(layout, cfg)
    windows = place_windows(layout, cfg, doors, np.random.default_rng(seed))
    openings = doors + windows

    def metric(bodies: list[Body]) -> list[Body]:
        return [(v * np.array([unit, unit, 1.0]), f) for v, f in bodies]

    # Walls: one horizontal slice per interval between opening heights; each slice is the
    # wall footprint minus the openings crossing it.
    levels = sorted({0.0, height, *(z for o in openings for z in (o.z0, o.z1) if 0 < z < height)})
    walls = Part("walls", tuple(colors["walls"]))
    for za, zb in pairwise(levels):
        cuts = [o.cut(layout) for o in openings if o.z0 <= za + 1e-9 and o.z1 >= zb - 1e-9]
        footprint = layout.walls.difference(unary_union(cuts)) if cuts else layout.walls
        walls.bodies += metric(extrude_all(footprint, za, zb))
    parts = [walls]

    outdoor = [k for k in range(len(layout.rooms)) if layout.is_outdoor(k)]
    slab_area = unary_union([layout.outer, *(layout.rooms[k] for k in outdoor)])
    slab = Part("slab", tuple(colors["slab"]))
    slab.bodies = metric(extrude_all(slab_area, -cfg["floor"]["slab_thickness"], 0.0))
    parts.append(slab)

    finish = cfg["floor"]["finish_thickness"]
    if finish > 0:
        floors: dict[str, Part] = {}
        for k, room in enumerate(layout.rooms):
            room_type = layout.types[k]
            area = room.difference(layout.outer if k in outdoor else layout.walls)
            part = floors.setdefault(
                room_type, Part(f"floor_{room_type}", tuple(colors["floors"][room_type]))
            )
            part.bodies += metric(extrude_all(area, 0.0, finish))
        parts += [p for p in floors.values() if p.bodies]

    door_part = Part("doors", tuple(colors["doors"]))
    for o in doors:
        if o.kind != "passage":
            panel = o.panel(layout, cfg["doors"]["panel_thickness"])
            door_part.bodies += metric(extrude_all(panel, o.z0, o.z1))
    glass = Part("glass", tuple(colors["glass"]))
    for o in windows:
        panel = o.panel(layout, cfg["windows"]["glass_thickness"])
        glass.bodies += metric(extrude_all(panel, o.z0, o.z1))

    railings = Part("railings", tuple(colors["railings"]))
    rail_t = cfg["balcony"]["railing_thickness"] / unit
    for k in outdoor:
        net = layout.rooms[k].difference(layout.outer)
        edge = layout.rooms[k].exterior.difference(layout.outer.buffer(0.01))
        strips = [p.buffer(rail_t / 2.0, cap_style="square") for p in straight_pieces(edge)]
        if strips:
            footprint = unary_union(strips).intersection(net)
            railings.bodies += metric(extrude_all(footprint, 0.0, cfg["balcony"]["railing_height"]))
    parts += [p for p in (door_part, glass, railings) if p.bodies]

    info = {
        "roof_requested": roof,
        "roof_built": roof,
        "roof_error": "",
        "pitch_deg": pitch,
        "doors": sum(o.kind != "passage" for o in doors),
        "windows": len(windows),
    }
    entrance = next((o for o in doors if o.kind == "entrance"), None)
    if entrance is not None:  # direction the entrance faces, used to aim the camera
        info["entrance_azimuth"] = round(
            float(np.degrees(np.arctan2(entrance.outward[1], entrance.outward[0]))), 1
        )
    if roof != "none":
        parts += _roof_parts(layout, cfg, roof, pitch, info)
    house = House(plan.id, parts, info)
    info["triangles"] = house.triangles()
    return house


def _roof_parts(layout, cfg: dict, roof: str, pitch: float, info: dict) -> list[Part]:
    rc, colors = cfg["roof"], cfg["materials"]
    height, unit = cfg["walls"]["height"], layout.unit
    footprints = [affinity.scale(p, unit, unit, origin=(0, 0)) for p in polygons(layout.outer)]
    tiles = Part("roof", tuple(colors["roof"]))
    gables = Part("gables", tuple(colors["walls"]))
    built = roof
    if built == "gable" and not all(
        is_nearly_rectangular(p, rc["gable_min_fill"]) for p in footprints
    ):
        built, info["roof_error"] = "hip", "footprint is not rectangular enough for a gable roof"
    if built == "gable":
        for p in footprints:
            slab, gable = gable_roof(p, height, pitch, rc["overhang"], rc["gable_thickness"])
            tiles.bodies.append(slab)
            gables.bodies.append(gable)
    if built == "hip":
        try:
            for p in footprints:
                body = hip_roof(p, height, pitch, rc["overhang"], rc["fascia"], rc["quantum"])
                problems = body_problems(body)
                if problems:
                    raise RoofError(f"invalid hip roof mesh: {', '.join(problems)}")
                tiles.bodies.append(body)
        except RoofError as error:
            built, info["roof_error"] = "flat", str(error)
            tiles.bodies = []
    if built == "flat":
        tiles = Part("roof", tuple(colors["roof_flat"]))
        for p in footprints:
            tiles.bodies += flat_roof(
                p, height, rc["flat_thickness"], rc["parapet_height"], rc["parapet_thickness"]
            )
    info["roof_built"] = built
    return [p for p in (tiles, gables) if p.bodies]
