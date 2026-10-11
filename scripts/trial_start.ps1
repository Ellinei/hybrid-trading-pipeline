<#
.SYNOPSIS
    Start or resume the forward paper trial on this PC.
.DESCRIPTION
    Starts Docker Desktop if needed, brings up Postgres and the Airflow
    scheduler, and runs one data refresh so the bot starts on current
    candles. Then it launches the bot detached and windowless: closing this
    window does not stop it, and the bot has no window of its own to close.
    Only scripts\trial_stop.ps1 stops it. The same paper account resumes;
    this script never creates a new one. Re-running is safe: it will not
    start a second bot.
.EXAMPLE
    pwsh -File scripts\trial_start.ps1
#>
[CmdletBinding()]
param(
    [int]$RefreshTimeoutSeconds = 900
)

. (Join-Path $PSScriptRoot 'trial_common.ps1')
Set-Location $Repo

if (Get-BotProcesses) {
    Write-Step 'The bot is already running. Use scripts\trial_status.ps1 to check it.'
    exit 0
}
if (-not (Test-Path $TrialDb)) {
    Write-Step "Trial account not found: $TrialDb. Refusing to create a new account."
    exit 1
}

# 1. Docker
if (-not (Test-DockerReady)) {
    Write-Step 'Starting Docker Desktop...'
    Start-Process $DockerDesktop
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Test-DockerReady)) {
        if ((Get-Date) -gt $deadline) { Write-Step 'Docker did not become ready within 4 minutes.'; exit 1 }
        Start-Sleep -Seconds 3
    }
}

# 2. Postgres + Airflow scheduler (airflow-init runs first as a dependency)
Write-Step 'Starting Postgres and the Airflow scheduler...'
docker compose up -d postgres airflow-scheduler
if ($LASTEXITCODE -ne 0) { Write-Step 'docker compose up failed.'; exit 1 }

$deadline = (Get-Date).AddMinutes(3)
while ($true) {
    $cid = docker compose ps -q postgres
    $health = if ($cid) { docker inspect -f '{{.State.Health.Status}}' $cid } else { '' }
    if ($health -eq 'healthy') { break }
    if ((Get-Date) -gt $deadline) { Write-Step "Postgres is not healthy (status: $health)."; exit 1 }
    Start-Sleep -Seconds 3
}

$deadline = (Get-Date).AddMinutes(3)
while ($true) {
    docker compose exec -T airflow-scheduler airflow dags unpause $DagId *> $null
    if ($LASTEXITCODE -eq 0) { break }
    if ((Get-Date) -gt $deadline) { Write-Step 'The Airflow CLI did not respond within 3 minutes.'; exit 1 }
    Start-Sleep -Seconds 5
}

# 3. Catch-up refresh: backfills every candle missed while the PC was off,
#    rebuilds and tests features, then publishes them.
$runId = 'trial_start__' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmss')
Write-Step "Refreshing candles and features (run $runId)..."
# A trigger fired right after the scheduler container starts can fail
# transiently, so retry until the run actually appears in the run list.
$registered = $false
$triggerOutput = @()
$deadline = (Get-Date).AddMinutes(3)
while (-not $registered -and (Get-Date) -lt $deadline) {
    $triggerOutput = docker compose exec -T airflow-scheduler airflow dags trigger $DagId --run-id $runId 2>&1
    $registered = [bool](docker compose exec -T airflow-scheduler airflow dags list-runs -d $DagId 2>$null |
        Select-String -SimpleMatch $runId)
    if (-not $registered) { Start-Sleep -Seconds 10 }
}
if (-not $registered) {
    Write-Step 'Could not trigger the refresh. Last trigger output:'
    $triggerOutput | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" }
}
$state = if ($registered) { '' } else { 'not triggered' }
$deadline = if ($registered) { (Get-Date).AddSeconds($RefreshTimeoutSeconds) } else { Get-Date }
while ((Get-Date) -lt $deadline) {
    $line = docker compose exec -T airflow-scheduler airflow dags list-runs -d $DagId 2>$null |
        Select-String -SimpleMatch $runId
    if ($line) { $state = ($line.Line -split '\|')[2].Trim() }
    if ($state -eq 'success' -or $state -eq 'failed') { break }
    Start-Sleep -Seconds 10
}
if ($state -eq 'success') {
    Write-Step 'Refresh succeeded. Latest candles and publication state:'
} else {
    Write-Step "WARNING: refresh state '$state'. The bot starts anyway; new buys stay blocked until a refresh succeeds (the hourly job retries at :05)."
}
Invoke-Psql 'SELECT symbol, max(timestamp) FROM raw.ohlcv GROUP BY symbol ORDER BY symbol;'
Invoke-Psql 'SELECT symbol, is_valid, feature_timestamp FROM raw.feature_validation ORDER BY symbol;'

# 4. Bot, detached through WMI so it is not a child of this window, and with
#    its console hidden. A visible console opens as a blank Windows Terminal
#    window titled cmd.exe (output goes to bot.log), and closing that window
#    kills the bot without a log line.
$logSizeBefore = if (Test-Path $BotLog) { (Get-Item $BotLog).Length } else { 0 }
$cmd = "cmd.exe /c `"cd /d `"$Repo`" && set BOT_OPERATIONS_DB=$TrialDb&& set PYTHONIOENCODING=utf-8&& .venv\Scripts\python.exe -u -m cli trade --paper --interval 3600 >> `"$BotLog`" 2>&1`""
$hidden = New-CimInstance -Namespace root/cimv2 -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = [ciminstance]$hidden
}
if ($result.ReturnValue -ne 0) { Write-Step "Launching the bot failed (WMI code $($result.ReturnValue))."; exit 1 }
$started = Get-UtcStamp
@(
    "cmd_wrapper=$($result.ProcessId)"
    "started_utc=$started"
    'stop: pwsh -File scripts\trial_stop.ps1'
) | Set-Content $PidFile
Add-Content $SessionsLog "start $started wrapper_pid=$($result.ProcessId)"

# 5. Confirm the first cycle ran.
$deadline = (Get-Date).AddSeconds(120)
$newLines = @()
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    $stream = [System.IO.File]::Open($BotLog, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$stream.Seek($logSizeBefore, 'Begin')
        $newLines = (New-Object System.IO.StreamReader($stream)).ReadToEnd() -split "`r?`n" | Where-Object { $_.Trim() }
    } finally { $stream.Dispose() }
    if ($newLines -match 'cycle_complete_sleeping|Traceback') { break }
}
$newLines | Select-Object -Last 6 | ForEach-Object { Write-Host "  $_" }
if ($newLines -match 'Traceback') { Write-Step 'The bot reported an error; see the log above.'; exit 1 }
if ($newLines -match 'cycle_complete_sleeping') {
    Write-Step 'Bot running. First cycle complete; next one in an hour.'
} else {
    Write-Step 'Bot launched, but the first cycle had not finished after 2 minutes. Check scripts\trial_status.ps1.'
}
