# Creates a Desktop shortcut that launches the app with the app icon and a
# minimized console window. Run once from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts/create-shortcut.ps1
#
# The shortcut runs uv.exe itself ("uv run streamlit run app.py", the same
# command as run.bat): Windows offers "Pin to taskbar" only for shortcuts to
# an .exe, so one to run.bat could not be pinned. Run this again if uv moves.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$desktop = [Environment]::GetFolderPath("Desktop")

$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $uv) {
    Write-Error "uv was not found on PATH - install it (see README), then run this again."
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop "Video & Audio Transcription.lnk"))
$lnk.TargetPath = $uv.Source
$lnk.Arguments = "run streamlit run app.py"
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7  # 7 = minimized (console starts minimized to the taskbar)
$lnk.IconLocation = Join-Path $root "assets\icon.ico"
$lnk.Description = "Start the Video & Audio Transcription app"
$lnk.Save()

Write-Host "Desktop shortcut created: Video & Audio Transcription (right-click it, Pin to taskbar)"
