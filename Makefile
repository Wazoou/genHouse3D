# Every target is a thin wrapper around a `uv run python -m ...` command, so the same
# commands work without make (e.g. on Windows): copy the recipe line.
DATA_CONFIG ?= configs/data/swiss_dwellings.yaml
# PyTorch flavour: cu126 (local GPU) or cpu (CI, CPU-only machines).
EXTRA ?= cu126
RUN = uv run --extra $(EXTRA)

.PHONY: data viz fixtures build3d render evaluate test lint format

data:  ## raw dataset -> unified plans, filters, de-duplication, frozen split
	$(RUN) python -m housegen.data.build --config $(DATA_CONFIG)

viz:  ## data report: reports/data_report.md + reports/figures/data/
	$(RUN) python -m housegen.data.report --config $(DATA_CONFIG)

fixtures:  ## refresh tests/fixtures/plans.jsonl from the built dataset
	$(RUN) python tests/fixtures/make_fixtures.py

build3d:  ## real plans -> GLB houses in outputs/, validation and roof statistics
	$(RUN) python -m housegen.geometry3d.cli --config $(DATA_CONFIG)

render:  ## gallery figures in reports/figures/3d/
	$(RUN) python -m housegen.render --config $(DATA_CONFIG)

evaluate:  ## metrics of the reference, the baselines and the model -> reports/results/
	$(RUN) python -m housegen.evaluate --config $(DATA_CONFIG)

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .
