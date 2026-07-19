@echo off
REM Umi-OCR 插件入口：用本插件内独立 venv 的 python 运行 engine.py
REM %~dp0 保证无论从哪个工作目录调用，都能定位到本插件目录
setlocal
set "DIR=%~dp0"
REM 优先使用 .venv_gpu（含 CUDA/GPU 支持），回退 .venv（纯 CPU）
if exist "%DIR%.venv_gpu\Scripts\python.exe" (
    set "VENV=.venv_gpu"
) else if exist "%DIR%.venv\Scripts\python.exe" (
    set "VENV=.venv"
) else (
    echo [Error] 未找到虚拟环境（.venv_gpu 或 .venv），请先运行 部署.bat 或按文档安装。
    exit /b 1
)
call "%DIR%%VENV%\Scripts\python.exe" "%DIR%engine.py" %*
endlocal
