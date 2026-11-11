# genHouse3D

Generating novel 3D houses from room graphs: a floorplan diffusion model followed by procedural
3D extrusion.

> **Status: work in progress.** Milestones 1 (data) and 2 (3D pipeline on real plans) of 8
> are done. The diffusion model is not built yet: every house below is extruded from a
> **real** plan of the dataset, not from a generated one.

![Houses extruded from real plans](reports/figures/3d/gallery_real_small.png)

## Pipeline (target)

room graph → diffusion model → 2D floorplan → geometric clean-up → 3D house (GLB)

## Data

Plans come from [Swiss Dwellings v3.0.0](https://doi.org/10.5281/zenodo.7788422): one plan per
single-storey apartment, rotated onto the axes, with rooms sharing edges on a 10 cm grid, a
door graph and a frozen train / val / test split by site. Counts, rejection reasons and
distributions are in the generated [data report](reports/data_report.md).

![20 random training plans](reports/figures/data/examples.png)

## 3D pipeline

`housegen.geometry3d` turns a plan into a house and exports it as GLB, in about 50 ms:

- **Walls**: 20 cm outer walls built outwards from the footprint, 10 cm partitions centred
  on shared room edges, none between rooms that are open onto each other.
- **Openings**: doors from the plan; windows on outer walls by room type (bays for the
  living room, a small high window for the bathroom, none for storage), seeded.
- **Roofs**: hip roof from the straight skeleton of the footprint, gable roof when the
  footprint is roughly rectangular, flat roof with parapet as fallback.
- **Cutaway**: `roof="none"` exports the house without its roof.

The straight skeleton is implemented in the repository for rectilinear footprints (see
[docs/decisions.md](docs/decisions.md)) and checked against a closed-form roof height.
Measured rates over the test split are in
[reports/results/roof_stats_real.csv](reports/results/roof_stats_real.csv); the full gallery
of 20 random test plans is [here](reports/figures/3d/gallery_real.png).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

Download `swiss-dwellings-v3.0.0.zip` (932 MB) from the Zenodo record and extract
`geometries.csv` to `data/raw/swiss_dwellings/swiss-dwellings-v3.0.0/`, then:

```bash
make data      # raw dataset -> unified plans, filters, split
make viz       # data report
make build3d   # 20 real plans -> GLB in outputs/, validation, roof statistics
make render    # gallery figures
make test
```

Each target wraps one `uv run python -m ...` command (see the [Makefile](Makefile)), which
can be run directly where `make` is not installed.

Design choices are logged in [docs/decisions.md](docs/decisions.md).

## Credits and licences

- **Data**: Standfest et al., *Swiss Dwellings: a large dataset of apartment models* (v3.0.0),
  Archilyse AG, [doi:10.5281/zenodo.7788422](https://doi.org/10.5281/zenodo.7788422),
  CC BY 4.0. The plans drawn in this repository and the test fixtures are derived from it
  (rotated, rectified, snapped to a grid, converted to the project format).
- **Model inspiration**: Shabani, Hosseini and Furukawa, *HouseDiffusion: Vector Floorplan
  Generation via a Diffusion Model with Discrete and Continuous Denoising*, CVPR 2023.
- **Libraries**: shapely, trimesh, numpy, networkx, matplotlib.
- **Code**: MIT.
