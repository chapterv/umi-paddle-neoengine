@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "MODE=install"
set "NO_PAUSE=0"
set "FORMULA_MODE=layout"
set "FORMULA_MODEL=PP-FormulaNet_plus-S"

:PARSE_ARGS
if "%~1"=="" goto ARGS_DONE
if /I "%~1"=="--help" goto HELP
if /I "%~1"=="--check" set "MODE=check"
if /I "%~1"=="--from-setup" set "NO_PAUSE=1"
if /I "%~1"=="--whole-image" set "FORMULA_MODE=whole_image"
if /I "%~1"=="--layout" set "FORMULA_MODE=layout"
if /I "%~1"=="--plus-m" set "FORMULA_MODEL=PP-FormulaNet_plus-M"
if /I "%~1"=="--plus-s" set "FORMULA_MODEL=PP-FormulaNet_plus-S"
shift
goto PARSE_ARGS

:HELP
echo install_formula_models.bat [--check] [--from-setup] [--layout] [--whole-image] [--plus-s] [--plus-m]
echo.
echo   默认          预下载 P1 公式模型（默认 plus-S，混排 layout）
echo   --check       只检查当前运行环境，不下载
echo   --from-setup  由 setup.bat 调用，不暂停窗口
echo   --layout      预下载混排模式所需模型（默认）
echo   --whole-image 仅预下载整图公式模型
echo   --plus-s      使用 PP-FormulaNet_plus-S（默认）
echo   --plus-m      使用 PP-FormulaNet_plus-M
exit /b 0

:ARGS_DONE
set "PLUGIN="
set "STATUS_PY="
if exist "%~dp0Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py\download_formula_models.py" (
  set "PLUGIN=%~dp0Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py"
)
if not defined PLUGIN if exist "%~dp0win_x64_PaddleOCR_Py\download_formula_models.py" (
  set "PLUGIN=%~dp0win_x64_PaddleOCR_Py"
)
if not defined PLUGIN (
  echo [ERROR] 找不到 win_x64_PaddleOCR_Py 或 download_formula_models.py。
  set "RC=2"
  goto FINISH
)
set "STATUS_PY=%PLUGIN%\install_status.py"

set "PY="
if defined TABLE_PY if exist "%TABLE_PY%" set "PY=%TABLE_PY%"
if defined FORMULA_PY if exist "%FORMULA_PY%" set "PY=%FORMULA_PY%"

if not defined PY if exist "%PLUGIN%\.venv_gpu\Scripts\python.exe" if exist "%PLUGIN%\.venv_gpu\Lib\site-packages\paddleocr" if exist "%PLUGIN%\.venv_gpu\Lib\site-packages\onnxruntime" (
  set "PY=%PLUGIN%\.venv_gpu\Scripts\python.exe"
)
if not defined PY if exist "%PLUGIN%\.venv\Scripts\python.exe" if exist "%PLUGIN%\.venv\Lib\site-packages\paddleocr" if exist "%PLUGIN%\.venv\Lib\site-packages\onnxruntime" (
  set "PY=%PLUGIN%\.venv\Scripts\python.exe"
)
if not defined PY if exist "%PLUGIN%\.venv_gpu\Scripts\python.exe" if exist "%PLUGIN%\.venv_gpu\Lib\site-packages\paddleocr" (
  set "PY=%PLUGIN%\.venv_gpu\Scripts\python.exe"
)
if not defined PY if exist "%PLUGIN%\.venv\Scripts\python.exe" if exist "%PLUGIN%\.venv\Lib\site-packages\paddleocr" (
  set "PY=%PLUGIN%\.venv\Scripts\python.exe"
)
if not defined PY (
  echo [ERROR] 未找到可用的插件虚拟环境。请先运行 setup.bat。
  set "RC=3"
  goto FINISH
)

echo ============================================================
echo  P1 公式模型安装/检查
echo ============================================================
echo [1] 插件目录：%PLUGIN%
echo [2] Python：%PY%
echo [3] 目标模式：%FORMULA_MODE%
echo [4] 目标模型：%FORMULA_MODEL%

if /I "%MODE%"=="check" (
  "%PY%" "%PLUGIN%\download_formula_models.py" --check
  set "RC=%ERRORLEVEL%"
  if "%RC%"=="0" if exist "%STATUS_PY%" "%PY%" "%STATUS_PY%" mark-optional --name formula_p1 --status checked --detail "download_formula_models --check ok" >nul 2>&1
  goto FINISH
)

echo [5] 预下载 P1 公式模型...
"%PY%" "%PLUGIN%\download_formula_models.py" --download --mode "%FORMULA_MODE%" --model-name "%FORMULA_MODEL%"
if errorlevel 1 (
  echo [ERROR] 公式模型预下载失败。基础 OCR 仍可继续使用。
  if exist "%STATUS_PY%" "%PY%" "%STATUS_PY%" mark-optional --name formula_p1 --status failed --detail "download_formula_models --download" --error "model predownload failed" >nul 2>&1
  set "RC=4"
  goto FINISH
)
echo [OK] P1 公式模型已就绪。
if exist "%STATUS_PY%" "%PY%" "%STATUS_PY%" mark-optional --name formula_p1 --status complete --detail "mode=%FORMULA_MODE%; model=%FORMULA_MODEL%" >nul 2>&1
set "RC=0"

:FINISH
if not defined RC set "RC=1"
if "%NO_PAUSE%"=="0" pause
exit /b %RC%
