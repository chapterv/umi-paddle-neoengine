@echo off
REM Umi-OCR 插件入口：用本插件内独立 venv 的 python 运行 engine.py
REM %~dp0 保证无论从哪个工作目录调用，都能定位到本插件目录
setlocal
set "DIR=%~dp0"
if not exist "%DIR%.venv\Scripts\python.exe" (
    echo [Error] 未找到 .venv 虚拟环境，请先按文档安装 paddlepaddle + paddleocr。
    exit /b 1
)
call "%DIR%.venv\Scripts\python.exe" "%DIR%engine.py" %*
endlocal
