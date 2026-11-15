@echo off
rem The app from the command line (cli.py), from any folder:
rem   scripts\v2t.bat transcribe "C:\Meetings\2026-11-05 14-03-25.mkv"
rem The project's own Python, not `uv run`: a plain `uv run` syncs first and
rem would uninstall the offline engine.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\cli.py" %*
