@echo off
chcp 65001 >nul
REM ============================================================
REM  输电线路智能路径规划系统 V5.0 —— 一键运行（台湾真实走廊验收）
REM  用法：修改下方 PY 与 DATA 路径后双击运行
REM ============================================================

SET PY=python
SET ROOT=%~dp0
SET DEM=%ROOT%data\taiwan_dem_35m.tif
SET DATA=%ROOT%..\data\real_cases_tw
SET OUT=%ROOT%results\v5_full

IF NOT EXIST "%DEM%" (
  echo [错误] 未找到 DEM 文件：%DEM%
  pause
  exit /b 1
)

IF NOT EXIST "%DATA%" (
  echo [错误] 未找到测试数据目录：%DATA%
  echo 请将 real_cases.csv 和对应的 *_real_path.shp 放入该目录。
  pause
  exit /b 1
)

IF NOT EXIST "%DATA%\real_cases.csv" (
  echo [错误] 未找到测试用例清单：%DATA%\real_cases.csv
  pause
  exit /b 1
)

echo [信息] 开始运行 VIN-Grad + PPO + A* 三方法集成规划 ...
echo [信息] 引导参数 real_attract=0.4  corridor_w=2.5  corridor_band=1200

pushd "%ROOT%"
%PY% -m transmission_path_planner ^
  --dem "%DEM%" ^
  --real-dir "%DATA%" ^
  --real-cases-csv "%DATA%\real_cases.csv" ^
  --out-dir "%OUT%" ^
  --guide-mode reference ^
  --real-attract 0.4 --corridor-w 2.5 --corridor-band 1200 ^
  --baseline
SET RUN_EXIT_CODE=%ERRORLEVEL%
popd

IF NOT "%RUN_EXIT_CODE%"=="0" (
  echo [错误] 路径规划运行失败，退出码：%RUN_EXIT_CODE%
  pause
  exit /b %RUN_EXIT_CODE%
)

echo [完成] 结果已写入 %OUT%
pause
