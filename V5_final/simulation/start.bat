@echo off
cd /d "%~dp0"
set "V5_PYTHON=D:\python\python.exe"
if exist "%V5_PYTHON%" (
  "%V5_PYTHON%" server.py %*
) else (
  python server.py %*
)
if errorlevel 1 pause
