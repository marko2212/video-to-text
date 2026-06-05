# Creates a Desktop shortcut that launches the app with the app icon and a
# minimized console window. Run once from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts/create-shortcut.ps1
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$desktop = [Environment]::GetFolderPath("Desktop")

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop "Video & Audio Transcription.lnk"))
$lnk.TargetPath = Join-Path $scriptDir "run.bat"
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7  # 7 = minimized (console starts minimized to the taskbar)
$lnk.IconLocation = Join-Path $root "assets\icon.ico"
$lnk.Description = "Start the Video & Audio Transcription app"
$lnk.Save()

Write-Host "Desktop shortcut created: Video & Audio Transcription"
