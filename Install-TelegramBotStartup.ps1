param(
    [string]$TaskName = "Long2Shorts Telegram Bot"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $projectRoot "Start-TelegramBot.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher was not found: $launcher"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Description "Private Long2Shorts Telegram control bot" -Force -ErrorAction Stop
    Write-Output "Registered scheduled task '$TaskName'. It starts at your next Windows sign-in."
}
catch {
    # Some managed Windows accounts cannot create Scheduled Tasks. The current
    # user's Startup folder provides the same logon-time behavior without admin
    # rights.
    $startupFolder = [Environment]::GetFolderPath("Startup")
    $startupLauncher = Join-Path $startupFolder "Long2Shorts Telegram Bot.vbs"
    $legacyCmd = Join-Path $startupFolder "Long2Shorts Telegram Bot.cmd"
    $command = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`""
    $quote = [string][char]34
    $escapedCommand = $command.Replace($quote, $quote + $quote)
    $vbs = 'CreateObject("Wscript.Shell").Run "' + $escapedCommand + '", 0, False'
    # UTF-16 VBS preserves project paths that contain Korean characters.
    Set-Content -LiteralPath $startupLauncher -Value $vbs -Encoding Unicode
    Remove-Item -LiteralPath $legacyCmd -Force -ErrorAction SilentlyContinue
    Write-Output "Scheduled Task permission was unavailable; added Startup launcher: $startupLauncher"
}
