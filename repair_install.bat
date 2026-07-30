@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PLUGIN=%~dp0Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py"
set "STATUS_PY=%PLUGIN%\install_status.py"
set "RUNPY="
set "RUNPY_ARGS="
set "AUTO_CHECK=0"

if /I "%~1"=="--help" goto HELP
if /I "%~1"=="--check" set "AUTO_CHECK=1"

if exist "%PLUGIN%\.venv_gpu\python.exe" set "RUNPY=%PLUGIN%\.venv_gpu\python.exe"
if not defined RUNPY if exist "%PLUGIN%\.venv_gpu\Scripts\python.exe" set "RUNPY=%PLUGIN%\.venv_gpu\Scripts\python.exe"
if not defined RUNPY if exist "%PLUGIN%\.venv\python.exe" set "RUNPY=%PLUGIN%\.venv\python.exe"
if not defined RUNPY if exist "%PLUGIN%\.venv\Scripts\python.exe" set "RUNPY=%PLUGIN%\.venv\Scripts\python.exe"
if not defined RUNPY py -3.11 --version >nul 2>&1 && set "RUNPY=py" && set "RUNPY_ARGS=-3.11"
if not defined RUNPY py -3.10 --version >nul 2>&1 && set "RUNPY=py" && set "RUNPY_ARGS=-3.10"
if not defined RUNPY python --version >nul 2>&1 && set "RUNPY=python"
if "%AUTO_CHECK%"=="1" goto CHECK

echo ============================================================
echo  Local-Ocr 检查并修复
echo ============================================================
echo   [1] 检查当前安装状态
echo   [2] 修复基础 OCR 环境（重跑 setup.bat）
echo   [3] 修复 P1 表格模型（install_table_models.bat）
echo   [4] 修复 P1 公式模型（install_formula_models.bat）
set /p ACTION=请输入 1/2/3/4（直接回车 = 1）：
if "%ACTION%"=="" set "ACTION=1"
if "%ACTION%"=="1" goto CHECK
if "%ACTION%"=="2" call "%~dp0setup.bat" & exit /b %ERRORLEVEL%
if "%ACTION%"=="3" call "%~dp0install_table_models.bat" & exit /b %ERRORLEVEL%
if "%ACTION%"=="4" call "%~dp0install_formula_models.bat" & exit /b %ERRORLEVEL%
echo [ERROR] 输入无效。
exit /b 2

:HELP
echo repair_install.bat [--check]
echo.
echo   默认       打开“检查并修复”菜单
echo   --check    仅输出 install_status.json 摘要
echo   修复入口   setup.bat（基础 OCR） / install_table_models.bat（P1 表格模型） / install_formula_models.bat（P1 公式模型）
exit /b 0

:CHECK
if not defined RUNPY (
  echo [ERROR] 未找到可用 Python，无法读取安装状态。
  echo         请先运行 setup.bat，或安装 Python 3.10/3.11 后重试。
  exit /b 3
)
if not exist "%STATUS_PY%" (
  echo [ERROR] 未找到 install_status.py：%STATUS_PY%
  exit /b 4
)
%RUNPY% %RUNPY_ARGS% "%STATUS_PY%" summary
exit /b %ERRORLEVEL%
