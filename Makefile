PY := .venv/bin/python
PORT ?= 4104

.PHONY: install fetch-converter seed run run-replay test test-slow test-live record-fixtures

install:            ## venv from the lockfile, with the Axolotl extra
	uv sync --frozen --all-extras

fetch-converter:    ## pinned llama.cpp GGUF converter into var/llama.cpp
	bash scripts/fetch_converter.sh

seed:               ## rebuild the synthetic dataset
	$(PY) -m coa seed

run:                ## web app on 127.0.0.1:$(PORT); live audit when a Claude login is found
	PORT=$(PORT) $(PY) -m coa serve

run-replay:         ## same app, audit and base-model answers replayed from fixtures/
	DEMO_REPLAY=1 PORT=$(PORT) $(PY) -m coa serve

test:               ## unit + end-to-end tests (replay mode, tiny model, no network once cached)
	DEMO_REPLAY=1 $(PY) -m pytest -q

test-slow:          ## axolotl preprocess and GGUF conversion checks
	$(PY) -m pytest -q -m slow

test-live:          ## one real audit call through the local Claude login
	LIVE=1 $(PY) -m pytest -q -m live

record-fixtures:    ## re-record replay fixtures from real live runs
	$(PY) scripts/record_fixtures.py
