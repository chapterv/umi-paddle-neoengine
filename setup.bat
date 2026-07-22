@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PLUGIN=Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py"
set "VENV=%PLUGIN%\.venv_gpu"
set "PY=%VENV%\Scripts\python.exe"
set "CHOICE_FILE=%TEMP%\local_ocr_choice.txt"

echo ============================================================
echo  Local-Ocr 新引擎部署（两步：模型范围 -^> 推理后端）
echo ============================================================

REM --- 1) 准备 Python（优先用 uv 自动下载 3.12，venv 内自带解释器）---
REM ⚠️ 根因（GPU 1.26/1.27 装不上）：
REM   onnxruntime-gpu 1.26+ 只有 cp311/cp312 wheel，**没有 cp310**。
REM   用 3.10 建 venv → 官方源也只到 1.23.2。故 GPU 部署强制目标 Python 3.12。
set "USE_UV=0"
set "PYBIN="
set "PYVER=0"
REM 与主项目可用 GPU 环境对齐：.venv_gpu 实测为 CPython 3.11 + ort-gpu 1.26.0
set "UV_PY=3.11"
where uv >nul 2>&1 && set "USE_UV=1"
if "%USE_UV%"=="1" (
  echo [1] 检测到 uv，按主项目方式安装/使用 CPython %UV_PY% 再建 venv...
  uv python install %UV_PY%
  if errorlevel 1 (
    echo [WARN] uv python install %UV_PY% 失败，尝试回退系统 Python...
    set "USE_UV=0"
  ) else (
    REM 解析 uv 管理的 python 绝对路径，供 menu.py 等直接调用
    for /f "delims=" %%P in ('uv python find %UV_PY% 2^>nul') do set "PYBIN=%%P"
    if defined PYBIN (
      set "PYVER=311"
      echo       uv python: !PYBIN!
    ) else (
      echo [WARN] uv python find 失败，回退系统 Python...
      set "USE_UV=0"
    )
  )
)
if not defined PYBIN (
  py -3.11 --version >nul 2>&1 && set "PYBIN=py -3.11" && set "PYVER=311"
)
if not defined PYBIN (
  py -3.12 --version >nul 2>&1 && set "PYBIN=py -3.12" && set "PYVER=312"
)
if not defined PYBIN if exist "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" (
  set "PYBIN=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
  set "PYVER=312"
)
if not defined PYBIN if exist "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe" (
  set "PYBIN=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
  set "PYVER=311"
)
if not defined PYBIN if exist "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" (
  set "PYBIN=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
  set "PYVER=311"
)
if not defined PYBIN (
  py -3.10 --version >nul 2>&1 && set "PYBIN=py -3.10" && set "PYVER=310"
)
if not defined PYBIN ( python --version >nul 2>&1 && set "PYBIN=python" && set "PYVER=0" )
if not defined PYBIN (
  echo [ERROR] 未找到可用 Python，且 uv 不可用。
  echo   方案A: 安装 uv 后重跑（推荐，会自动拉 Python 3.12）
  echo          https://docs.astral.sh/uv/getting-started/installation/
  echo   方案B: 手动安装 Python 3.11/3.12 并勾选 Add to PATH
  pause
  exit /b 1
)
echo [1] 使用 Python: %PYBIN%  ^(tag=%PYVER%, uv=%USE_UV%^)
if "%PYVER%"=="310" (
  echo [WARN] 当前为 Python 3.10：onnxruntime-gpu 最高仅 1.23.2，装不了 1.26/1.27。
  echo        请安装 uv 或 Python 3.11+ 后重跑。
)

REM --- 2) 第1步：模型范围（复用 menu.py）---
echo [2] 第1步：选择预下载的模型范围
if exist "%CHOICE_FILE%" del /q "%CHOICE_FILE%" >nul 2>&1
%PYBIN% menu.py "%CHOICE_FILE%"
set "CHOICE=2"
if exist "%CHOICE_FILE%" (
  for /f "usebackq delims=" %%L in ("%CHOICE_FILE%") do set "CHOICE=%%L"
)
echo     -^> 已选模型范围：%CHOICE%

REM --- 3) 第2步：推理后端 ---
echo.
echo 第2步：选择推理后端
echo   [1] 纯 GPU   : 仅装 onnxruntime-gpu（需 N 卡 + 驱动）
echo   [2] 纯 CPU   : 仅装 onnxruntime（默认，体积最小）
echo   [3] 全安装   : GPU + CPU 两种后端都装
set /p BACKEND=请输入 1/2/3（直接回车 = 选项2）：
if "%BACKEND%"=="" set BACKEND=2

set "SPEC="
set "SPEC_EXTRA="
set "FULL_INSTALL=0"
if "%BACKEND%"=="2" (
  REM 纯 CPU：装进 .venv（不是 .venv_gpu），并跳过第3步 GPU/CUDA 选择。
  REM 关键：VENV 改变时必须同步更新 PY（指向新 venv 的 python.exe），
  REM 否则 PY 仍指向 .venv_gpu 的 python → 后续 if not exist "%PY%" 误判失败。
  REM CPU 与主项目对齐：Python 3.11+ 用 1.26.0；3.10 最高 1.23.2。
  set "VENV=%PLUGIN%\.venv"
  set "PY=%PLUGIN%\.venv\Scripts\python.exe"
  if "%PYVER%"=="310" (
    set "SPEC=onnxruntime==1.23.2"
  ) else (
    set "SPEC=onnxruntime==1.26.0"
  )
  goto INSTALL
)
if "%BACKEND%"=="3" (
  REM 全安装：.venv_gpu 装 onnxruntime-gpu（内含 CPU EP，可同时服务 onnxruntime 与 gpu 引擎）
  REM 额外再装一份纯 CPU 到 .venv，避免 run.cmd 只命中残缺 GPU 环境。
  set "FULL_INSTALL=1"
)

REM --- 4) 第3步（仅 GPU 类后端）：按 Python 版本给出「真实可装」的 SPEC ---
REM cp310: 最高 1.23.2（无 1.26/1.27 wheel）
REM cp311+:
REM   · 1.26.0[cuda,cudnn] → 依赖 nvidia-*-cu12，PyPI 齐全（推荐，与主项目一致）
REM   · 1.27.0[cuda,cudnn] → 依赖 nvidia-cuda-nvrtc-cu13~=13.0 等；
REM     实测 PyPI 上 cu13 包只有占位版 0.0.1，**无真实 13.x wheel**，
REM     故 [cuda,cudnn] 会失败——这与 Python 3.11 无关！
REM     选项1 改为：先装 onnxruntime-gpu==1.27.0 本体；extras 能装再装，装不上不硬崩。
set "GPU_WANT_CUDA13=0"
set "SPEC_BASE="
echo.
if "%PYVER%"=="310" (
  echo 第3步：GPU onnxruntime 版本（Python 3.10 限制）
  echo   [1] onnxruntime-gpu==1.23.2  （本解释器唯一可选的最新档）
  echo       说明：1.26/1.27 无 cp310 轮子；要 CUDA12.9/13 请改用 Python 3.11+ 重跑。
  set /p CUDAV=请输入 1（直接回车 = 1）：
  if "!CUDAV!"=="" set CUDAV=1
  if "!CUDAV!"=="1" (
    set "SPEC=onnxruntime-gpu==1.23.2"
    set "SPEC_BASE=onnxruntime-gpu==1.23.2"
  )
) else (
  echo 第3步：选择 GPU 的 onnxruntime + CUDA 版本（Python 3.11+）
  echo   [1] onnxruntime-gpu 1.27.0（CUDA 13 目标）
  echo       注意：PyPI 尚无可用的 nvidia-*-cu13 正式包，[cuda,cudnn] 会装失败；
  echo       脚本会装 1.27 本体；CUDA EP 需本机已有 CUDA 13 运行库，否则自动回退 CPU。
  echo   [2] onnxruntime-gpu 1.26.0 + CUDA 12.9  pip 自带 DLL  （默认·推荐）
  echo       （RTX 30/40 等；nvidia-*-cu12 可从 PyPI 自动装齐，无需系统 CUDA Toolkit）
  set /p CUDAV=请输入 1/2（直接回车 = 选项2）：
  if "!CUDAV!"=="" set CUDAV=2
  if "!CUDAV!"=="1" (
    REM 本体可装；extras 单独尝试（见下方 INSTALL）
    set "SPEC=onnxruntime-gpu==1.27.0"
    set "SPEC_BASE=onnxruntime-gpu==1.27.0"
    set "GPU_WANT_CUDA13=1"
  )
  if "!CUDAV!"=="2" (
    set "SPEC=onnxruntime-gpu[cuda,cudnn]==1.26.0"
    set "SPEC_BASE=onnxruntime-gpu==1.26.0"
  )
)
if not defined SPEC (
  echo 输入无效，已取消。
  pause
  exit /b 1
)
echo     -^> GPU 将安装：%SPEC%
if "%GPU_WANT_CUDA13%"=="1" echo     -^> （随后尝试 cu13 extras；失败则保留本体，不中断部署）

:INSTALL
echo.
REM 判断是否有「完整」venv（python.exe + pip.exe 都在）。残缺/上次中断留下的空目录
REM 会让 "if not exist %VENV%" 误判为已存在而跳过创建，最终 python 检查失败。
REM 另：若旧 .venv_gpu 是 3.10 建的，装不了 1.26/1.27 → 检测 major.minor，不匹配则重建。
set "VENV_OK=0"
if exist "%VENV%\Scripts\python.exe" if exist "%VENV%\Scripts\pip.exe" set "VENV_OK=1"
if "%VENV_OK%"=="1" (
  for /f "delims=" %%V in ('"%VENV%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')" 2^>nul') do set "VENV_PYVER=%%V"
  if defined PYVER if not "%PYVER%"=="0" if not "!VENV_PYVER!"=="%PYVER%" (
    echo [info] 已有 venv 的 Python 版本 !VENV_PYVER! 与当前选择 %PYVER% 不一致，删除后按新版本重建...
    rmdir /s /q "%VENV%" >nul 2>&1
    set "VENV_OK=0"
  )
)
if "%VENV_OK%"=="0" (
  if exist "%VENV%" (
    echo [info] 发现残缺/未完成的虚拟环境 %VENV%，删除后重建...
    rmdir /s /q "%VENV%" >nul 2>&1
  )
  echo [3] 创建虚拟环境 %VENV% ...
  if "%USE_UV%"=="1" (
    REM uv venv 会把指定 CPython 写进 venv，后续 pip 都走 venv 内 python
    uv venv --python %UV_PY% --seed --clear "%VENV%"
  ) else (
    %PYBIN% -m venv "%VENV%"
  )
)
if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] venv 创建失败，未生成 python.exe。
  echo         若有 uv: uv python install 3.12 ^&^& uv venv --python 3.12 "%VENV%"
  echo         或手动: py -3.12 -m venv "%VENV%"
  pause & exit /b 1
)
for /f "delims=" %%V in ('"%VENV%\Scripts\python.exe" -c "import sys; print(sys.version.split()[0])" 2^>nul') do echo       venv python = %%V

REM pip 国内镜像（更快 + 避免国际 TLS 中断）；覆盖：set PIP_INDEX=https://pypi.org/simple
if not defined PIP_INDEX set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_EXTRA=--index-url %PIP_INDEX% --trusted-host pypi.tuna.tsinghua.edu.cn --retries 10 --timeout 60"

"%PY%" -m pip install --quiet --upgrade pip %PIP_EXTRA%
if errorlevel 1 ( echo [WARN] pip 自升级失败，继续。 )
echo        安装 paddlepaddle + paddleocr ...
"%PY%" -m pip install "paddlepaddle==3.2.1" "paddleocr==3.7.0" %PIP_EXTRA%
if errorlevel 1 (
  echo [ERROR] paddlepaddle/paddleocr 安装失败，检查网络/代理后重试。
  pause
  exit /b 1
)
echo        安装推理后端 %SPEC% ...
"%PY%" -m pip install "%SPEC%" %PIP_EXTRA%
if errorlevel 1 (
  echo [WARN] 镜像 %PIP_INDEX% 未同步 %SPEC%，自动改用官方源重试（可能较慢）...
  "%PY%" -m pip install "%SPEC%" --index-url https://pypi.org/simple --retries 10 --timeout 120
)
if errorlevel 1 (
  REM 带 [cuda,cudnn] 失败时：改装同版本本体（cu13 extras 在 PyPI 无正式包时必走此路）
  if defined SPEC_BASE if /I not "%SPEC%"=="%SPEC_BASE%" (
    echo [WARN] 带 CUDA extras 的安装失败，改装不带 extras 的本体 %SPEC_BASE% ...
    "%PY%" -m pip install "%SPEC_BASE%" --index-url https://pypi.org/simple --retries 10 --timeout 120
    if not errorlevel 1 (
      echo [WARN] 已装上 %SPEC_BASE%（无 pip 自带 CUDA DLL）。
      echo        · 推荐：重跑 setup，第3步选 [2] 1.26.0[cuda,cudnn]（cu12 可一键装齐）
      echo        · 若坚持 1.27：需本机自备 CUDA 运行库，或等 nvidia-*-cu13 上架 PyPI
      set "SPEC=%SPEC_BASE%"
      goto ORT_INSTALLED
    )
  )
  echo [ERROR] 推理后端 %SPEC% 安装失败。
  echo   这通常**不是** Python 3.11 的问题。请对照日志：
  echo   A^) 若提示 nvidia-cuda-nvrtc-cu13 / nvidia-*-cu13：PyPI 尚无 cu13 正式包
  echo      → 请重跑 setup，第3步选 [2] onnxruntime-gpu 1.26.0 + CUDA 12.9（推荐）
  echo   B^) 若是 Python 3.10：确实装不了 1.26/1.27 — 请用 3.11+ 删 .venv_gpu 重跑
  echo   C^) 网络/镜像：set PIP_INDEX=https://pypi.org/simple 后重试
  echo   注意：若只装了 paddleocr 而没装上 onnxruntime，Umi 默认 engine=onnxruntime 会 OCR init fail。
  pause
  exit /b 1
)
:ORT_INSTALLED

REM CUDA13 目标：本体已装时，额外尝试 [cuda,cudnn]；失败不阻断（PyPI 常无 cu13）
if "%GPU_WANT_CUDA13%"=="1" (
  echo        尝试为 1.27 安装 CUDA13 extras: onnxruntime-gpu[cuda,cudnn]==1.27.0 ...
  "%PY%" -m pip install "onnxruntime-gpu[cuda,cudnn]==1.27.0" --index-url https://pypi.org/simple --retries 5 --timeout 90
  if errorlevel 1 (
    echo [WARN] CUDA13 extras 不可用（预期现象：nvidia-*-cu13 在 PyPI 只有占位 0.0.1）。
    echo        已保留 onnxruntime-gpu==1.27.0 本体 → import 正常；GUI 的 onnxruntime 引擎可用。
    echo        CUDAExecutionProvider 仅在本机已有 CUDA13 运行库时可能启用，否则 engine 自动回退 CPU。
    echo        若要「pip 一键带齐 CUDA DLL」：请重跑 setup 选 [2] 1.26.0 + CUDA12.9。
  ) else (
    echo        CUDA13 extras 安装成功。
  )
)

REM 强制校验：GUI 默认 engine=onnxruntime，import 必须成功（onnxruntime-gpu 也提供同名模块）
echo        校验 import onnxruntime ...
"%PY%" -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
if errorlevel 1 (
  echo [ERROR] onnxruntime 已 pip 安装但 import 失败。请删掉 %VENV% 后重跑 setup.bat。
  pause
  exit /b 1
)

REM 全安装：再准备一份 CPU .venv（含 onnxruntime），避免只剩残缺 .venv_gpu 时无法用默认引擎
if "%FULL_INSTALL%"=="1" (
  echo        [全安装] 额外创建/更新 CPU 环境 .venv ...
  set "VENV_CPU=%PLUGIN%\.venv"
  set "PY_CPU=%PLUGIN%\.venv\Scripts\python.exe"
  set "VENV_CPU_OK=0"
  if exist "!VENV_CPU!\Scripts\python.exe" if exist "!VENV_CPU!\Scripts\pip.exe" set "VENV_CPU_OK=1"
  if "!VENV_CPU_OK!"=="0" (
    if exist "!VENV_CPU!" rmdir /s /q "!VENV_CPU!" >nul 2>&1
    if "%USE_UV%"=="1" (
      uv venv --python %UV_PY% --seed --clear "!VENV_CPU!"
    ) else (
      %PYBIN% -m venv "!VENV_CPU!"
    )
  )
  if exist "!PY_CPU!" (
    "!PY_CPU!" -m pip install --quiet --upgrade pip %PIP_EXTRA%
    "!PY_CPU!" -m pip install "paddlepaddle==3.2.1" "paddleocr==3.7.0" %PIP_EXTRA%
    if "%PYVER%"=="310" (
      "!PY_CPU!" -m pip install "onnxruntime==1.23.2" %PIP_EXTRA%
    ) else (
      "!PY_CPU!" -m pip install "onnxruntime==1.26.0" %PIP_EXTRA%
    )
    if errorlevel 1 (
      "!PY_CPU!" -m pip install "onnxruntime==1.26.0" --index-url https://pypi.org/simple --retries 10 --timeout 120
    )
    "!PY_CPU!" -c "import onnxruntime as o; print('[全安装] CPU ort', o.__version__)"
    if errorlevel 1 (
      echo [WARN] CPU .venv 的 onnxruntime 校验失败；.venv_gpu 仍可用（若 GPU 包完整）。
    ) else (
      echo        [全安装] CPU .venv 就绪。
    )
  ) else (
    echo [WARN] 未能创建 CPU .venv，跳过全安装的 CPU 半部。
  )
)

echo        预下载模型（范围选项 %CHOICE%）...
"%PY%" "%~dp0deploy.py" --choice %CHOICE%
if errorlevel 1 ( echo [WARN] 模型预下载异常，首次识别会自动下载。 )

echo.
echo ============================================================
echo  部署完成。现在可以：
echo   1. 双击 Umi-OCR\Umi-OCR.exe 打开主程序
echo   2. 在「全局设置 - 文字识别」选择 PaddleOCR（本地·PP-OCRv6/v5/v4）
echo   3. 拖入图片开始 - 首次会自动用对应引擎
echo ============================================================
pause
goto :eof
