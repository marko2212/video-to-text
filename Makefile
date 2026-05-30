.DEFAULT_GOAL := help

.PHONY: help run sync lock upgrade upgrade-package lint format fix test check clean clean-venv reset

PYTHON_NO_PROJECT := uv run --no-project python

help:
	@echo Available targets:
	@echo   make run                         - Start Streamlit app
	@echo   make sync                        - Install dependencies from uv.lock
	@echo   make lock                        - Refresh uv.lock without upgrading
	@echo   make upgrade                     - Upgrade all deps to latest allowed versions
	@echo   make upgrade-package PKG=openai  - Upgrade one package and sync
	@echo   make lint                        - Run ruff check
	@echo   make format                      - Run ruff format
	@echo   make fix                         - Auto-fix lint issues
	@echo   make test                        - Run the pytest suite
	@echo   make check                       - Lint + format check + tests (CI gate)
	@echo   make clean                       - Remove temp/, uploads/ contents and __pycache__
	@echo   make clean-venv                  - Remove .venv
	@echo   make reset                       - clean-venv + sync

run:
	uv run streamlit run app.py

sync:
	uv sync

lock:
	uv lock

upgrade:
	uv lock --upgrade
	uv sync

upgrade-package:
	uv lock --upgrade-package $(PKG)
	uv sync

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

reset: clean-venv sync
