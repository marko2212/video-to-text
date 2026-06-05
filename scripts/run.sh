#!/usr/bin/env bash
# Linux/macOS launcher. Run from a terminal:  bash scripts/run.sh
# (On macOS you can copy this to run.command to make it double-clickable.)
cd "$(dirname "$0")/.." || exit 1
exec uv run streamlit run app.py
