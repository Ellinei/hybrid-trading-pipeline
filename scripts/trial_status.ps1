<#
.SYNOPSIS
    Read-only status of the forward paper trial.
.EXAMPLE
    pwsh -File scripts\trial_status.ps1
#>
. (Join-Path $PSScriptRoot 'trial_common.ps1')
Set-Location $Repo

Write-Host '== Bot =='
$procs = @(Get-BotProcesses)
if ($procs.Count -gt 0) {
    $first = ($procs | Sort-Object CreationDate | Select-Object -First 1).CreationDate.ToUniversalTime()
    Write-Host "Running since $($first.ToString('yyyy-MM-dd HH:mm')) UTC ($($procs.Count) python processes)"
} else {
    Write-Host 'NOT running. Start it with: pwsh -File scripts\trial_start.ps1'
}

Write-Host "`n== Services =="
$dbUp = $false
if (Test-DockerReady) {
    foreach ($svc in 'postgres', 'airflow-scheduler') {
        $state = if (Test-ServiceRunning $svc) { 'running' } else { 'stopped' }
        Write-Host "${svc}: $state"
    }
    $dbUp = Test-ServiceRunning 'postgres'
} else {
    Write-Host 'Docker is not running.'
}

if ($dbUp) {
    Write-Host "`n== Data (latest candle, publication) =="
    Invoke-Psql 'SELECT symbol, max(timestamp) FROM raw.ohlcv GROUP BY symbol ORDER BY symbol;'
    Invoke-Psql 'SELECT symbol, is_valid, feature_timestamp FROM raw.feature_validation ORDER BY symbol;'
    if (Test-ServiceRunning 'airflow-scheduler') {
        Write-Host "`n== Last refresh runs =="
        docker compose exec -T airflow-scheduler airflow dags list-runs -d $DagId 2>$null |
            Where-Object { $_ -match $DagId } | Select-Object -First 3 |
            ForEach-Object { $f = $_ -split '\|'; Write-Host ("{0}  {1}" -f $f[2].Trim(), $f[1].Trim()) }
    }
}

Write-Host "`n== Trial =="
if ($dbUp) { & $Python (Join-Path $PSScriptRoot 'trial_report.py') }
else { & $Python (Join-Path $PSScriptRoot 'trial_report.py') --no-db }

if (Test-Path $SessionsLog) {
    Write-Host "`n== Recent sessions =="
    Get-Content $SessionsLog -Tail 6
}
if (Test-Path $BotLog) {
    Write-Host "`n== Last log lines =="
    Get-Content $BotLog -Tail 4 | Where-Object { $_.Trim() }
}
