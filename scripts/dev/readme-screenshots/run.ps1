<#
.SYNOPSIS
  Retake the README screenshots and demo GIF (docs/screenshots/) on made-up recordings.

.DESCRIPTION
  Makes a demo meeting with Windows text-to-speech (two minutes of talk, then a
  recorder left running) and four more recordings, copies the app's tracked
  files into -Out (no .env, so no API key; its own data, temp and upload
  folders), starts that copy on 127.0.0.1:<Port> with OpenAI pointed nowhere,
  and drives it with puppeteer: the folder list, a Local run of the trimmed
  meeting, a history of made-up rows next to that run, and the same flow
  again as frames for demo.gif. The real data/, temp/ and uploads/ are never
  read or written. See README.md here.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dev\readme-screenshots\run.ps1
#>
param(
  [string]$Out = (Join-Path $env:TEMP 'video-to-text-screenshots'),
  [int]$Port = 8597,
  [int]$TrimEnd = 100,
  [switch]$NoVideo,  # only the three screenshots, not docs/screenshots/demo.gif
  [switch]$Keep      # leave the demo instance running (to rerun one shoot.mjs step)
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here '..\..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$demo = Join-Path $here 'demo.py'
$shoot = Join-Path $here 'shoot.mjs'
$url = "http://127.0.0.1:$Port/"
$marker = Join-Path $Out '.readme-screenshots'

function Invoke-Native([string]$What, [scriptblock]$Block) {
  Write-Host "== $What"
  # Windows PowerShell 5.1 turns a native program's stderr into errors when
  # its output is captured; judge by the exit code instead.
  $ErrorActionPreference = 'Continue'
  $global:LASTEXITCODE = 0
  & $Block
  if ($LASTEXITCODE) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Stop-Instance {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'streamlit' -and $_.CommandLine -match "--server\.port $Port(\s|$)" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

Stop-Instance  # one left running by -Keep
Start-Sleep -Milliseconds 500
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
  throw "Port $Port is in use - stop what runs there or pass -Port."
}
if (Test-Path $Out) {
  if (-not (Test-Path $marker)) { throw "$Out exists and was not made by this script." }
  Remove-Item -Recurse -Force $Out
}
New-Item -ItemType Directory -Force $Out | Out-Null
New-Item -ItemType File $marker | Out-Null
foreach ($d in 'tts', 'inst\data', 'inst\temp', 'inst\uploads') {
  New-Item -ItemType Directory -Force (Join-Path $Out $d) | Out-Null
}

Write-Host '== speech (Windows text-to-speech)'
Add-Type -AssemblyName System.Speech
$lines = @(
  @('Zira', 'Good morning everyone, and thanks for joining. This is our sprint planning for the next two weeks. Let us start with what we finished last sprint.'),
  @('David', 'Sure. The subtitle export is done and released. We also fixed the bug where long recordings stopped halfway, so a run that loses its connection now picks up where it left off.'),
  @('Zira', 'Great. What about the screenshots feature?'),
  @('David', 'That one is almost ready. The app now takes a screenshot only when the picture on the screen changes, so a long call with one shared slide costs a few cents instead of a dollar.'),
  @('Zira', 'Good. For this sprint, I would like us to focus on three things. First, the history page should load faster. Second, we need clearer messages when the account runs out of credit. And third, the documentation needs new screenshots.'),
  @('David', 'I can take the history page. We can load each transcript only when someone opens it.'),
  @('Zira', 'Perfect. I will write the messages, and I will ask design about the screenshots. Any risks?'),
  @('David', 'Only one. The release on Friday depends on the new build server, and it is not set up yet.'),
  @('Zira', 'Okay, let us check on that tomorrow. Thanks everyone, that is it for today.')
)
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
for ($i = 0; $i -lt $lines.Count; $i++) {
  $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $voice.SelectVoice("Microsoft $($lines[$i][0]) Desktop")
  $voice.SetOutputToWaveFile((Join-Path $Out ('tts\line{0:D2}.wav' -f $i)), $format)
  $voice.Speak($lines[$i][1])
  $voice.SetOutputToNull()
  $voice.Dispose()
}

if (Test-Path 'C:\ffmpeg\bin') { $env:PATH = 'C:\ffmpeg\bin;' + $env:PATH }
Invoke-Native 'recordings' { & $python $demo media $Out }
# The folder list shows when a recording was started: the file's creation time.
$times = Get-Content -Raw -Encoding UTF8 (Join-Path $Out 'times.json') | ConvertFrom-Json
foreach ($entry in $times.PSObject.Properties) {
  $file = Get-Item -LiteralPath (Join-Path $Out "Recordings\$($entry.Name)")
  $file.CreationTime = [datetime]$entry.Value.created
  $file.LastWriteTime = [datetime]$entry.Value.saved
}
Invoke-Native 'app copy' { & $python $demo copy-app $repo (Join-Path $Out 'app') }

# The instance: its own folders, no key, OpenAI unreachable, models from the repo.
$app = Join-Path $Out 'app'
$env:DATA_DIR = Join-Path $Out 'inst\data'
$env:TEMP_DIR = Join-Path $Out 'inst\temp'
$env:UPLOAD_DIR = Join-Path $Out 'inst\uploads'
$env:WHISPER_MODEL_DIR = Join-Path $repo 'models'
$env:OPENAI_API_KEY = ''
$env:OPENAI_BASE_URL = 'http://127.0.0.1:9'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $app
Push-Location $app
try {
  Invoke-Native 'preferences' { & $python $demo prefs (Join-Path $Out 'Recordings') }
  Write-Host "== starting the demo instance on $url"
  Start-Process -FilePath $python -WorkingDirectory $app -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Out 'instance.log') -RedirectStandardError (Join-Path $Out 'instance.err') `
    -ArgumentList '-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', "$Port", `
    '--server.headless', 'true', '--client.toolbarMode', 'minimal', '--browser.gatherUsageStats', 'false' | Out-Null
  $deadline = (Get-Date).AddSeconds(90)
  do {
    Start-Sleep -Milliseconds 500
    try { $up = (Invoke-WebRequest -UseBasicParsing "$($url)_stcore/health" -TimeoutSec 2).StatusCode -eq 200 } catch { $up = $false }
  } until ($up -or (Get-Date) -gt $deadline)
  if (-not $up) { throw "The instance did not start - see $Out\instance.err" }

  Invoke-Native 'home.png' { node $shoot $url home (Join-Path $Out 'home.png') }
  Invoke-Native 'transcribe.png (a Local run)' { node $shoot $url transcribe (Join-Path $Out 'transcribe.png') $TrimEnd }
  Invoke-Native 'history rows' { & $python $demo history }
  Invoke-Native 'history.png' { node $shoot $url history (Join-Path $Out 'history.png') }
  Invoke-Native 'crop into docs/screenshots' { & $python $demo crop $Out (Join-Path $repo 'docs\screenshots') }
  if (-not $NoVideo) {
    # After history.png: the video's run adds one more row to the demo history.
    $video = Join-Path $Out 'video'
    New-Item -ItemType Directory -Force $video | Out-Null
    Invoke-Native 'demo video (another Local run)' { node $shoot $url video (Join-Path $video 'demo.json') $TrimEnd }
    Invoke-Native 'demo.gif' { & $python $demo gif (Join-Path $video 'demo.json') (Join-Path $repo 'docs\screenshots\demo.gif') }
  }
}
finally {
  Pop-Location
  if ($Keep) { Write-Host "The demo instance keeps running on $url (-Keep); the next run of this script stops it." }
  else { Stop-Instance }
}
Write-Host "Done. Check docs\screenshots\, then 'git diff --stat'. Work folder: $Out"
