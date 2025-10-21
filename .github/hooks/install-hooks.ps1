# Windows PowerShell script to install Git hooks
# Usage: .\.github\hooks\install-hooks.ps1

$ErrorActionPreference = "Stop"

Write-Host "Installing Git Hooks..." -ForegroundColor Cyan

# Check if in Git repository
if (-not (Test-Path ".git")) {
    Write-Host "Error: Not in Git repository root directory" -ForegroundColor Red
    exit 1
}

# Copy pre-commit hook
$sourceHook = ".github/hooks/pre-commit"
$targetHook = ".git/hooks/pre-commit"

if (Test-Path $sourceHook) {
    Copy-Item $sourceHook $targetHook -Force
    Write-Host "pre-commit hook installed to .git/hooks/" -ForegroundColor Green
} else {
    Write-Host "Error: Cannot find $sourceHook" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Git Hooks installation complete!" -ForegroundColor Green
Write-Host "The following will run before each commit:" -ForegroundColor Yellow
Write-Host "  - ruff check (code linting)" -ForegroundColor Gray
Write-Host "  - black --check (format check)" -ForegroundColor Gray
Write-Host "  - pytest (run tests)" -ForegroundColor Gray
