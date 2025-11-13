@echo off
REM 运行剩余的5个S1实验
REM 在第一个实验(seed=42)完成后运行此脚本

echo ====================================
echo S1 剩余实验自动训练脚本
echo 共5个实验，每个约2小时
echo ====================================
echo.

echo [%date% %time%] 开始训练序列
echo.

echo ====================================
echo 2/6: S1 IID seed=43
echo ====================================
python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=43
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: S1 IID seed=43 失败
) else (
    echo SUCCESS: S1 IID seed=43 完成
)
echo.

echo ====================================
echo 3/6: S1 IID seed=44
echo ====================================
python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=44
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: S1 IID seed=44 失败
) else (
    echo SUCCESS: S1 IID seed=44 完成
)
echo.

echo ====================================
echo 4/6: S1 Brand-OOD seed=42
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: S1 Brand-OOD seed=42 失败
) else (
    echo SUCCESS: S1 Brand-OOD seed=42 完成
)
echo.

echo ====================================
echo 5/6: S1 Brand-OOD seed=43
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: S1 Brand-OOD seed=43 失败
) else (
    echo SUCCESS: S1 Brand-OOD seed=43 完成
)
echo.

echo ====================================
echo 6/6: S1 Brand-OOD seed=44
echo ====================================
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: S1 Brand-OOD seed=44 失败
) else (
    echo SUCCESS: S1 Brand-OOD seed=44 完成
)
echo.

echo ====================================
echo 所有训练完成！
echo 完成时间: %date% %time%
echo ====================================
pause
