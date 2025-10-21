# PowerShell equivalent of Makefile for Windows users
# Usage: .\Makefile.ps1 <command>
# Example: .\Makefile.ps1 init

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

function Init {
    Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
    python -m pip install -U pip
    pip install -r requirements.txt
    Write-Host "Done!" -ForegroundColor Green
}

function Install-Hooks {
    Write-Host "Installing Git hooks..." -ForegroundColor Cyan
    if (Test-Path ".github/hooks/install-hooks.ps1") {
        & .\.github\hooks\install-hooks.ps1
    } else {
        Write-Host "Error: install-hooks.ps1 not found" -ForegroundColor Red
        exit 1
    }
}

function Lint {
    Write-Host "Running linters..." -ForegroundColor Cyan
    python -m ruff check .
    python -m black --check .
}

function Test {
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest -q
}

function Train {
    Write-Host "Starting training..." -ForegroundColor Cyan
    $env:HF_LOCAL_ONLY = "1"
    $env:HF_CACHE_DIR = Join-Path $PWD "models\roberta-base"
    $env:DATA_ROOT = Join-Path $PWD "data\processed"
    python scripts\train.py --profile local
}

function Eval {
    Write-Host "Running evaluation..." -ForegroundColor Cyan
    $env:HF_LOCAL_ONLY = "1"
    $env:HF_CACHE_DIR = Join-Path $PWD "models\roberta-base"
    $env:DATA_ROOT = Join-Path $PWD "data\processed"
    python scripts\train.py --profile server --eval-only
}

function Dvc-Init {
    Write-Host "Initializing DVC..." -ForegroundColor Cyan
    dvc init -q
    dvc remote add -d local ./dvcstore 2>$null
}

function Dvc-Track {
    Write-Host "Tracking data with DVC..." -ForegroundColor Cyan
    dvc add data/processed 2>$null
    git add data/processed.dvc .gitignore
}

function Dvc-Push {
    Write-Host "Pushing DVC data..." -ForegroundColor Cyan
    dvc push
}

function Validate-Data {
    Write-Host "Validating data schema..." -ForegroundColor Cyan
    python scripts\validate_data_schema.py
}

function Show-Help {
    Write-Host @"
PowerShell Makefile for Windows

Available commands:
  init            Install Python dependencies
  install-hooks   Install Git hooks
  lint            Run code linters (ruff + black)
  test            Run pytest
  train           Start training (local profile)
  eval            Run evaluation (server profile)
  dvc-init        Initialize DVC
  dvc-track       Track data with DVC
  dvc-push        Push DVC data
  validate-data   Validate data schema
  help            Show this help message

Usage:
  .\Makefile.ps1 <command>

Example:
  .\Makefile.ps1 init
  .\Makefile.ps1 lint
"@ -ForegroundColor Yellow
}

# Execute command
switch ($Command.ToLower()) {
    "init"          { Init }
    "install-hooks" { Install-Hooks }
    "lint"          { Lint }
    "test"          { Test }
    "train"         { Train }
    "eval"          { Eval }
    "dvc-init"      { Dvc-Init }
    "dvc-track"     { Dvc-Track }
    "dvc-push"      { Dvc-Push }
    "validate-data" { Validate-Data }
    "help"          { Show-Help }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\Makefile.ps1 help' for available commands" -ForegroundColor Yellow
        exit 1
    }
}

