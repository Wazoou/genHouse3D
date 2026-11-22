# Every target is a thin wrapper around a `uv run python -m ...` command, so the same
# commands work without make (e.g. on Windows): copy the recipe line.
DATA_CONFIG ?= configs/data/swiss_dwellings.yaml
# PyTorch flavour: cu126 (local GPU) or cpu (CI, CPU-only machines).
EXTRA ?= cu126
RUN = uv run --extra $(EXTRA)

.PHONY: data viz fixtures build3d render train sample overfit resume-check kaggle-bundle experiments evaluate test lint format

data:  ## raw dataset -> unified plans, filters, de-duplication, frozen split
	$(RUN) python -m housegen.data.build --config $(DATA_CONFIG)

viz:  ## data report: reports/data_report.md + reports/figures/data/
	$(RUN) python -m housegen.data.report --config $(DATA_CONFIG)

fixtures:  ## refresh tests/fixtures/plans.jsonl from the built dataset
	$(RUN) python tests/fixtures/make_fixtures.py

build3d:  ## real plans -> GLB houses in outputs/, validation and roof statistics
	$(RUN) python -m housegen.geometry3d.cli --config $(DATA_CONFIG)

render:  ## gallery of real plans (reports/figures/3d/) and of 24 random generated houses
	$(RUN) python -m housegen.render --config $(DATA_CONFIG)
	$(RUN) python -m housegen.gallery --checkpoint $(CHECKPOINT)

MODEL_CONFIG ?= configs/model/base.yaml
CHECKPOINT ?= runs/base/ema.pt

train:  ## train (or resume) the diffusion model; add AMP=--amp for fp16
	$(RUN) python -m housegen.train --config $(MODEL_CONFIG) $(AMP)

sample:  ## 5 plans per test graph with DDIM -> outputs/samples/
	$(RUN) python -m housegen.sample --checkpoint $(CHECKPOINT) --out outputs/samples/test_ddim.jsonl.gz

kaggle-bundle:  ## code.zip + plans.jsonl.gz in outputs/kaggle/, to upload as private datasets
	mkdir -p outputs/kaggle
	git archive --format=zip -o outputs/kaggle/code.zip HEAD
	cp data/processed/swiss_dwellings/plans.jsonl.gz outputs/kaggle/

overfit:  ## sanity check: memorise 32 plans, then regenerate them
	$(RUN) python -m housegen.overfit --config configs/model/overfit.yaml

resume-check:  ## kill a training run, relaunch it and compare with an uninterrupted run
	$(RUN) python -m housegen.resume_check --config configs/model/overfit.yaml

experiments:  ## sample, clean and evaluate the trained models (before / after clean-up)
	$(RUN) python -m housegen.experiments --model base=runs/base/ema.pt --variant no_rounding=base

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
