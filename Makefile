.DEFAULT_GOAL := help

.PHONY: help run sync sync-local lock upgrade upgrade-package lint format fix test check clean clean-venv reset

# --isolated: a python outside .venv, so clean-venv can delete .venv on Windows
# (a running .venv python cannot delete itself and left a broken venv behind).
PYTHON_NO_PROJECT := uv run --no-project --isolated python

# `uv sync` is exact: it uninstalls every optional extra it was not asked for, so
# a plain sync silently removed the offline engine. Keep the `local` extra when
# it is installed, or when downloaded models show it is wanted. Evaluated once,
# before `reset` deletes .venv. `make sync EXTRAS=` drops it for that one call;
# to drop it for good, also delete models/.
EXTRAS := $(if $(wildcard .venv/Lib/site-packages/faster_whisper .venv/lib/python3*/site-packages/faster_whisper models/models--*),--extra local,)

help:
	@echo Available targets:
	@echo   make run                         - Start Streamlit app
	@echo   make sync                        - Install dependencies from uv.lock, keeping the offline engine if installed
	@echo   make sync-local                  - Install dependencies plus the offline engine
	@echo   make lock                        - Refresh uv.lock without upgrading
	@echo   make upgrade                     - Upgrade all deps to latest allowed versions
	@echo   make upgrade-package PKG=openai  - Upgrade one package and sync
	@echo   make lint                        - Run ruff check
	@echo   make format                      - Run ruff format
	@echo   make fix                         - Auto-fix lint issues
	@echo   make test                        - Run the pytest suite
	@echo   make check                       - Lint + format check + tests, the CI gate
	@echo   make clean                       - Remove temp/, uploads/ contents and __pycache__
	@echo   make clean-venv                  - Remove .venv
	@echo   make reset                       - clean-venv + sync

run:
	uv run streamlit run app.py

sync:
	uv sync $(EXTRAS)

sync-local:
	uv sync --extra local

lock:
	uv lock

upgrade:
	uv lock --upgrade
	uv sync $(EXTRAS)

upgrade-package:
	uv lock --upgrade-package $(PKG)
	uv sync $(EXTRAS)

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .

test:
	uv run pytest

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

clean:
	$(PYTHON_NO_PROJECT) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(d, ignore_errors=True) or pathlib.Path(d).mkdir(exist_ok=True) for d in ('temp', 'uploads')]"

clean-venv:
	$(PYTHON_NO_PROJECT) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"

# Sequential even under make -j: the sync must not start before the delete ends.
reset:
	$(MAKE) clean-venv
	$(MAKE) sync EXTRAS="$(EXTRAS)"
