#!/usr/bin/env sh
# The app from the command line (cli.py), from any folder:
#   scripts/v2t.sh transcribe "$HOME/Meetings/2026-11-05 14-03-25.mkv"
# The project's own Python, not `uv run`: a plain `uv run` syncs first and
# would uninstall the offline engine.
root="$(cd "$(dirname "$0")/.." && pwd)"
python="$root/.venv/bin/python"
[ -x "$python" ] || python="$root/.venv/Scripts/python.exe"
exec "$python" "$root/cli.py" "$@"
