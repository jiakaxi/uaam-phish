@echo off
REM 运行3个S1 Brand-OOD实验

REM 确保在正确的目录
cd /d D:\uaam-phish
if not exist "scripts\train_hydra.py" (
    echo ERROR: 未找到 train_hydra.py，请确认在正确的目录
    pause
    exit /b 1
)

REM 设置PYTHONPATH确保模块能正确导入
set PYTHONPATH=%CD%

echo ====================================
echo S1 Brand-OOD 实验 (3 seeds)
echo 工作目录: %CD%
echo PYTHONPATH: %PYTHONPATH%
echo ====================================
echo.

echo [%date% %time%] 开始训练...
echo.

echo ====================================
echo 1/3: S1 Brand-OOD seed=42
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: seed=42 失败
) else (
    echo SUCCESS: seed=42 完成
)
echo.

echo ====================================
echo 2/3: S1 Brand-OOD seed=43
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: seed=43 失败
) else (
    echo SUCCESS: seed=43 完成
)
echo.

echo ====================================
echo 3/3: S1 Brand-OOD seed=44
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: seed=44 失败
) else (
    echo SUCCESS: seed=44 完成
)
echo.

echo ====================================
echo 所有Brand-OOD实验完成！
echo 完成时间: %date% %time%
echo ====================================
pause
