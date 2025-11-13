@echo off
cd /d D:\uaam-phish
set PYTHONPATH=%CD%

echo ====================================
echo Brand-OOD remaining experiments
echo ====================================
echo.

echo [%date% %time%] seed=43 starting...
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43
if %ERRORLEVEL% EQU 0 (
    echo [OK] seed=43 completed
) else (
    echo [ERROR] seed=43 failed
)
echo.

echo [%date% %time%] seed=44 starting...
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44
if %ERRORLEVEL% EQU 0 (
    echo [OK] seed=44 completed
) else (
    echo [ERROR] seed=44 failed
)
echo.

echo ====================================
echo All remaining experiments completed
echo ====================================
