@echo off
REM Umi-OCR 插件入口：用本插件内独立 venv 的 python 运行 engine.py
REM %~dp0 保证无论从哪个工作目录调用，都能定位到本插件目录
setlocal
set "DIR=%~dp0"
REM 选择 venv：优先 .venv_gpu（含 CUDA/GPU 支持），回退 .venv（纯 CPU）。
REM ⚠️ 只检查 python.exe 不够：.venv_gpu 可能存在 python.exe 但 paddleocr 未装/损坏
REM   （如 setup.bat GPU 分支 onnxruntime-gpu 装失败时 paddle 可能也没装上，
REM    或 .venv_gpu 被部分重建），导致 engine.py 在 `from paddleocr import PaddleOCR`
REM   处 ImportError -> Umi-OCR 报 "OCR init fail"。
REM   修复：额外检查 paddleocr 包目录是否存在，不存在则回退另一个 venv。
set "VENV="
if exist "%DIR%.venv_gpu\Lib\site-packages\paddleocr" (
    set "VENV=.venv_gpu"
)
if not defined VENV if exist "%DIR%.venv\Lib\site-packages\paddleocr" (
    set "VENV=.venv"
)
if not defined VENV (
    echo [Error] 未找到含 paddleocr 的虚拟环境（.venv_gpu 或 .venv），请先运行部署脚本。
    exit /b 1
)
call "%DIR%%VENV%\Scripts\python.exe" "%DIR%engine.py" %*
endlocal
