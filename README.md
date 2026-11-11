# genHouse3D

Generating novel 3D houses from room graphs: a floorplan diffusion model followed by procedural
3D extrusion.

> **Status: work in progress.** Milestone 1 of 8 (data) is done. The diffusion model, the 3D
> pipeline, the evaluation and the demo are not built yet, so there are no results to show.

## Pipeline (target)

room graph → diffusion model → 2D floorplan → geometric clean-up → 3D house (GLB)

## Data

Plans come from [Swiss Dwellings v3.0.0](https://doi.org/10.5281/zenodo.7788422): one plan per
single-storey apartment, rotated onto the axes, with rooms sharing edges on a 10 cm grid, a
door graph and a frozen train / val / test split by site. Counts, rejection reasons and
distributions are in the generated [data report](reports/data_report.md).

![20 random training plans](reports/figures/data/examples.png)

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

Download `swiss-dwellings-v3.0.0.zip` (932 MB) from the Zenodo record and extract
`geometries.csv` to `data/raw/swiss_dwellings/swiss-dwellings-v3.0.0/`, then:

```bash
make data   # uv run python -m housegen.data.build --config configs/data/swiss_dwellings.yaml
make viz    # uv run python -m housegen.data.report --config configs/data/swiss_dwellings.yaml
make test   # uv run pytest
```

Design choices are logged in [docs/decisions.md](docs/decisions.md).

## Credits and licences

- **Data**: Standfest et al., *Swiss Dwellings: a large dataset of apartment models* (v3.0.0),
  Archilyse AG, [doi:10.5281/zenodo.7788422](https://doi.org/10.5281/zenodo.7788422),
  CC BY 4.0. The plans drawn in this repository and the test fixtures are derived from it
  (rotated, rectified, snapped to a grid, converted to the project format).
- **Model inspiration**: Shabani, Hosseini and Furukawa, *HouseDiffusion: Vector Floorplan
  Generation via a Diffusion Model with Discrete and Continuous Denoising*, CVPR 2023.
- **Code**: MIT.
