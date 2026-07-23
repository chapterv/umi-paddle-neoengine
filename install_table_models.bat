@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "MODE=install"
set "NO_PAUSE=0"

:PARSE_ARGS
if "%~1"=="" goto ARGS_DONE
if /I "%~1"=="--help" goto HELP
if /I "%~1"=="--check" set "MODE=check"
if /I "%~1"=="--deps-only" set "MODE=deps"
if /I "%~1"=="--from-setup" set "NO_PAUSE=1"
shift
goto PARSE_ARGS

:HELP
echo install_table_models.bat [--check] [--deps-only] [--from-setup]
echo.
echo   默认          安装 P1 可选依赖并预下载表格结构模型
echo   --check       只检查当前运行环境，不安装、不下载
echo   --deps-only   只安装依赖；模型在首次开启 P1 时下载
echo   --from-setup  由 setup.bat 调用，不暂停窗口
exit /b 0

:ARGS_DONE
set "PLUGIN="
if exist "%~dp0Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py\requirements-table.txt" (
  set "PLUGIN=%~dp0Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py"
)
if not defined PLUGIN if exist "%~dp0win_x64_PaddleOCR_Py\requirements-table.txt" (
  set "PLUGIN=%~dp0win_x64_PaddleOCR_Py"
)
if not defined PLUGIN (
  echo [ERROR] 找不到 win_x64_PaddleOCR_Py 或 requirements-table.txt。
  set "RC=2"
  goto FINISH
)

set "PY="
if defined TABLE_PY if exist "%TABLE_PY%" set "PY=%TABLE_PY%"

REM 与插件 run.cmd 保持一致：优先完整 .venv_gpu，再回退完整 .venv。
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
echo  P1 表格结构模型安装/检查
echo ============================================================
echo [1] 插件目录：%PLUGIN%
echo [2] Python：%PY%

if /I "%MODE%"=="check" (
  "%PY%" "%PLUGIN%\download_table_models.py" --check
  set "RC=%ERRORLEVEL%"
  goto FINISH
)

if not defined PIP_INDEX set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_EXTRA=--index-url %PIP_INDEX% --trusted-host pypi.tuna.tsinghua.edu.cn --retries 10 --timeout 120"
echo [3] 安装 P1 可选依赖...
"%PY%" -m pip install -r "%PLUGIN%\requirements-table.txt" %PIP_EXTRA%
if errorlevel 1 (
  echo [WARN] 镜像安装失败，切换到 PyPI 官方源重试...
  "%PY%" -m pip install -r "%PLUGIN%\requirements-table.txt" --index-url https://pypi.org/simple --retries 10 --timeout 180
)
if errorlevel 1 (
  echo [ERROR] P1 可选依赖安装失败。
  set "RC=4"
  goto FINISH
)

"%PY%" "%PLUGIN%\download_table_models.py" --check
if errorlevel 1 (
  echo [ERROR] P1 依赖校验失败。
  set "RC=5"
  goto FINISH
)

if /I "%MODE%"=="deps" (
  echo [OK] P1 依赖已安装；表格模型会在首次开启 P1 时自动下载。
  set "RC=0"
  goto FINISH
)

echo [4] 预下载 P1 表格结构模型（约 955 MB，请耐心等待）...
"%PY%" "%PLUGIN%\download_table_models.py" --download
if errorlevel 1 (
  echo [ERROR] 表格模型预下载失败。依赖已经安装，可稍后重试本脚本。
  set "RC=6"
  goto FINISH
)
echo [OK] P1 表格结构模型已就绪。
set "RC=0"

:FINISH
if not defined RC set "RC=1"
if "%NO_PAUSE%"=="0" pause
exit /b %RC%
