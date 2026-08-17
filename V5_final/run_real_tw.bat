@echo off
REM ============================================================================
REM  真实输电走廊对齐评测 (v5: VIN-Grad + PPO + A* 三算法)
REM  目标: 规划线 vs 真实线 误差<=8%
REM  正确运行方式: 双击本文件 或 cmd /c "D:\大创\V5_final\run_real_tw.bat"
REM  不要直接把本文件内容复制粘贴到 cmd 窗口!
REM ============================================================================
set PY=D:\python\python.exe
set SCRIPT=D:\大创\V5_final\scripts\ai_path_planning_v5.py
set CSV=D:\大创\V5_final\data\real_cases_tw\real_cases.csv
set DEM=D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif
set OUT=D:\大创\V5_final\results\v5_real_tw_run

if not exist "%PY%" (
    echo [ERR] 找不到 Python: %PY%
    echo 请修改本文件第 9 行的 set PY=... 为你的 python.exe 路径
    pause
    exit /b 1
)

echo [CHECK] 检查 torch ...
"%PY%" -c "import torch" 2>nul
if errorlevel 1 (
    echo [INSTALL] 未检测到 torch, 正在自动安装 CUDA 11.8 版 torch ...
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [WARN] CUDA 版安装失败, 尝试 CPU 版 ...
        "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    )
    "%PY%" -c "import torch" 2>nul
    if errorlevel 1 (
        echo [WARN] torch 仍未安装成功, 本次跳过 VIN-Grad/PPO, 仅运行 A* 基线.
        set EXTRA=--no-ppp
    ) else (
        echo [OK] torch 安装成功.
        set EXTRA=
    )
) else (
    echo [OK] torch 已安装.
    set EXTRA=
)

echo [RUN] 启动评测 ...
"%PY%" "%SCRIPT%" ^
  --real-cases-csv "%CSV%" ^
  --dem "%DEM%" ^
  --out-dir "%OUT%" ^
  --real-attract 0.4 ^
  --corridor-w 2.5 ^
  --corridor-band 1200 %EXTRA%

set RC=errorlevel
echo.
echo 完成. 结果见 %OUT%\real_comparison_metrics.json
echo 若 PPO 崩溃, 请用文本编辑器打开本文件, 在末尾 %%EXTRA%% 位置直接加 --no-ppp
pause
exit /b %RC%