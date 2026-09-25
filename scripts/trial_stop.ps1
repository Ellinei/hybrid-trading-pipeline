<#
.SYNOPSIS
    Stop the forward paper trial cleanly before shutting the PC down.
.DESCRIPTION
    Waits for the bot to finish its current cycle (up to 90 s), stops it,
    records the stop time, and warns about open paper positions: their
    stops and time limits are not checked while the PC is off. Then it
    stops the Airflow scheduler and Postgres, unless -KeepServices is set.
.EXAMPLE
    pwsh -File scripts\trial_stop.ps1
#>
[CmdletBinding()]
param(
    [switch]$KeepServices
)

. (Join-Path $PSScriptRoot 'trial_common.ps1')
Set-Location $Repo

$procs = @(Get-BotProcesses)
if ($procs.Count -eq 0) {
    Write-Step 'The bot is not running.'
} else {
    # Stop between cycles so an evaluation is never cut off midway.
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $last = Get-Content $BotLog -Tail 5 | Where-Object { $_.Trim() } | Select-Object -Last 1
        if ($last -match 'cycle_complete_sleeping') { break }
        Write-Step 'Waiting for the current cycle to finish...'
        Start-Sleep -Seconds 5
    }
    # Kill each process tree from its root: the python.exe whose parent is
    # not another bot process (its cmd.exe wrapper goes with it).
    $ids = $procs.ProcessId
    foreach ($root in ($procs | Where-Object { $ids -notcontains $_.ParentProcessId })) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($root.ParentProcessId)"
        $target = if ($parent -and $parent.Name -eq 'cmd.exe') { $parent.ProcessId } else { $root.ProcessId }
        taskkill /PID $target /T /F *> $null
    }
    Start-Sleep -Seconds 2
    Get-BotProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    if (Get-BotProcesses) { Write-Step 'Some bot processes are still running; check Task Manager.'; exit 1 }
    $stopped = Get-UtcStamp
    Add-Content $SessionsLog "stop  $stopped"
    Write-Step 'Bot stopped.'
}

& $Python (Join-Path $PSScriptRoot 'trial_report.py') --open-only
Write-Host '  (Open positions are not monitored while the PC is off. The first cycle after'
Write-Host '   trial_start.ps1 closes any that expired and checks stops at the current price.)'

if (-not $KeepServices -and (Test-DockerReady)) {
    Write-Step 'Stopping the Airflow scheduler and Postgres...'
    docker compose stop airflow-scheduler postgres
}
Write-Step 'Done. It is now safe to shut down. Resume with: pwsh -File scripts\trial_start.ps1'
