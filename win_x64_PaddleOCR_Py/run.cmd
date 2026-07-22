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
REM 优先选「paddleocr + onnxruntime 都齐全」的环境。
REM 根因（部署 OCR init fail）：.venv_gpu 里只有 paddleocr、推理后端装失败/中断，
REM GUI 默认 engine=onnxruntime → paddlex 报 dependency not installed。
REM onnxruntime 与 onnxruntime-gpu 都会在 site-packages 下生成 onnxruntime 目录。
if exist "%DIR%.venv_gpu\Lib\site-packages\paddleocr" if exist "%DIR%.venv_gpu\Lib\site-packages\onnxruntime" (
    set "VENV=.venv_gpu"
)
if not defined VENV if exist "%DIR%.venv\Lib\site-packages\paddleocr" if exist "%DIR%.venv\Lib\site-packages\onnxruntime" (
    set "VENV=.venv"
)
REM 回退：仅有 paddleocr（可走 engine=paddle；ONNX 会在 engine 内给出明确错误）
if not defined VENV if exist "%DIR%.venv_gpu\Lib\site-packages\paddleocr" (
    set "VENV=.venv_gpu"
)
if not defined VENV if exist "%DIR%.venv\Lib\site-packages\paddleocr" (
    set "VENV=.venv"
)
if not defined VENV (
    echo [Error] 未找到含 paddleocr 的虚拟环境（.venv_gpu 或 .venv），请先运行部署脚本。
    exit /b 1
)
if not exist "%DIR%%VENV%\Lib\site-packages\onnxruntime" (
    echo [Warn] %VENV% 缺少 onnxruntime；若 GUI 推理引擎为 onnxruntime 将 init 失败。
    echo        请重跑 setup.bat 完成「推理后端」安装，或 pip install onnxruntime / onnxruntime-gpu。
)
call "%DIR%%VENV%\Scripts\python.exe" "%DIR%engine.py" %*
endlocal
