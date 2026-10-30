@echo off
rem Windows launcher. Double-click for a normal window, or use the Desktop
rem shortcut (create-shortcut.ps1): the same command, started minimized, and
rem it can be pinned to the taskbar.
rem A browser tab opens automatically.
cd /d "%~dp0.."
echo Starting the transcription app... a browser tab will open shortly.
uv run streamlit run app.py
