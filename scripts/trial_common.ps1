# Shared settings for the forward paper-trial scripts. Dot-source only:
#   . (Join-Path $PSScriptRoot 'trial_common.ps1')
# Native commands (docker) report failure through $LASTEXITCODE, which every
# caller checks explicitly. 'Continue' keeps redirected stderr from aborting
# the script under Windows PowerShell 5.1.
$ErrorActionPreference = 'Continue'

$Repo          = Split-Path -Parent $PSScriptRoot
$TrialDir      = Join-Path $Repo 'results\forward_20260924'
$TrialDb       = Join-Path $Repo 'results\forward_20260924.sqlite3'
$BotLog        = Join-Path $TrialDir 'bot.log'
$PidFile       = Join-Path $TrialDir 'bot.pid'
$SessionsLog   = Join-Path $TrialDir 'sessions.log'
$Python        = Join-Path $Repo '.venv\Scripts\python.exe'
$DagId         = 'ohlcv_feature_refresh'
$DockerDesktop = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'

function Get-UtcStamp { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }

function Write-Step([string]$Message) { Write-Host "[$(Get-UtcStamp)] $Message" }

# The venv launcher and `cli trade` each spawn a child, so one bot shows up as
# several python.exe processes. Any match means the bot is running.
function Get-BotProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'cli trade --paper|execution\.bot_runner' }
}

function Test-DockerReady {
    docker info *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-ServiceRunning([string]$Service) {
    $id = docker compose ps -q --status running $Service 2>$null
    return [bool]$id
}

function Invoke-Psql([string]$Sql) {
    docker compose exec -T postgres psql -U trader -d trading -tA -F ' | ' -c $Sql 2>$null
}
