$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python was not found: $python"
}

$monitor = Join-Path $projectRoot "trend_monitor.py"
$existingMonitor = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "^python(\.exe)?$" -and $_.CommandLine -match "trend_monitor\.py"
}
if (-not $existingMonitor -and (Test-Path -LiteralPath $monitor)) {
    Start-Process -FilePath $python -ArgumentList @($monitor) -WindowStyle Hidden
}

& $python (Join-Path $projectRoot "telegram_bot.py")
