@echo off
rem Windows launcher. Double-click for a normal window, or use the Desktop
rem shortcut (create-shortcut.ps1) to start it minimized.
rem A browser tab opens automatically.
cd /d "%~dp0.."
echo Starting the transcription app... a browser tab will open shortly.
uv run streamlit run app.py
