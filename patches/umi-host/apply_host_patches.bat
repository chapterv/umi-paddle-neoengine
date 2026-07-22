@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================
REM  Umi 宿主补丁一键部署（v1.1）
REM  将本目录 5 个 py 覆盖到 Umi-OCR 的 py_src（批量文档假死等修复）
REM
REM  用法：
REM    1) 双击本 bat（自动在常见位置查找 Umi-OCR）
REM    2) 拖拽「Umi-OCR」文件夹 / 「UmiOCR-data」文件夹到本 bat 上
REM    3) 命令行：apply_host_patches.bat "D:\path\to\Umi-OCR"
REM
REM  注意：请先完全退出 Umi-OCR（含托盘），再执行。
REM ============================================================

set "PATCH_DIR=%~dp0"
set "TARGET=%~1"

echo.
echo === umi-paddle-neoengine 宿主补丁部署 ===
echo 补丁目录: %PATCH_DIR%
echo.

REM --- 校验补丁文件齐全 ---
set "MISS=0"
for %%F in (
  mission.py
  mission_doc.py
  mission_ocr.py
  BatchDOC.py
  line_preprocessing.py
) do (
  if not exist "%PATCH_DIR%%%F" (
    echo [错误] 缺少补丁文件: %%F
    set "MISS=1"
  )
)
if "!MISS!"=="1" (
  echo 请从完整源码仓 patches\umi-host 运行本脚本。
  goto :fail
)

REM --- 解析目标 Umi 路径 ---
if not "%TARGET%"=="" goto :resolve_target

REM 无参数：按常见相对路径猜测
set "CAND="
REM 1) 发布包结构：本仓库上级的 Local-Ocr / 解压包
for %%P in (
  "%PATCH_DIR%..\..\..\Local-Ocr\Umi-OCR"
  "%PATCH_DIR%..\..\Umi-OCR"
  "%PATCH_DIR%..\..\..\Umi-OCR"
  "%PATCH_DIR%..\..\..\umi-cpu\Umi-OCR"
  "C:\tools\Umi-Ocr"
  "C:\tools\Umi-OCR"
) do (
  if exist "%%~fP\UmiOCR-data\py_src\mission\mission.py" (
    set "CAND=%%~fP"
    goto :got_cand
  )
  if exist "%%~fP\py_src\mission\mission.py" (
    set "CAND=%%~fP"
    goto :got_cand
  )
)

:got_cand
if defined CAND (
  set "TARGET=!CAND!"
  echo [自动] 找到 Umi: !TARGET!
  goto :resolve_target
)

echo 未找到 Umi-OCR。请将「Umi-OCR」文件夹拖到本 bat 上，或：
echo   apply_host_patches.bat "D:\你的路径\Umi-OCR"
echo.
goto :fail

:resolve_target
REM 去掉尾部反斜杠
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET:~0,-1%"

set "PY_SRC="
if exist "%TARGET%\UmiOCR-data\py_src\mission\mission.py" (
  set "PY_SRC=%TARGET%\UmiOCR-data\py_src"
  goto :do_copy
)
if exist "%TARGET%\py_src\mission\mission.py" (
  set "PY_SRC=%TARGET%\py_src"
  goto :do_copy
)
if exist "%TARGET%\mission\mission.py" (
  set "PY_SRC=%TARGET%"
  goto :do_copy
)
if exist "%TARGET%\Umi-OCR\UmiOCR-data\py_src\mission\mission.py" (
  set "PY_SRC=%TARGET%\Umi-OCR\UmiOCR-data\py_src"
  goto :do_copy
)

echo [错误] 路径无法识别为 Umi-OCR / UmiOCR-data / py_src:
echo   %TARGET%
echo 需要其下存在 py_src\mission\mission.py
goto :fail

:do_copy
echo 目标 py_src: %PY_SRC%
echo.

REM 备份时间戳
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "BAK=%PY_SRC%\_patch_backup_%TS%"
mkdir "%BAK%" 2>nul
mkdir "%BAK%\mission" 2>nul
mkdir "%BAK%\tag_pages" 2>nul
mkdir "%BAK%\parser_tools" 2>nul

echo [备份] %BAK%
copy /Y "%PY_SRC%\mission\mission.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_doc.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_ocr.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\tag_pages\BatchDOC.py" "%BAK%\tag_pages\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" "%BAK%\parser_tools\" >nul 2>&1

echo [覆盖] 写入补丁...
copy /Y "%PATCH_DIR%mission.py" "%PY_SRC%\mission\mission.py" >nul || goto :copy_fail
copy /Y "%PATCH_DIR%mission_doc.py" "%PY_SRC%\mission\mission_doc.py" >nul || goto :copy_fail
copy /Y "%PATCH_DIR%mission_ocr.py" "%PY_SRC%\mission\mission_ocr.py" >nul || goto :copy_fail
copy /Y "%PATCH_DIR%BatchDOC.py" "%PY_SRC%\tag_pages\BatchDOC.py" >nul || goto :copy_fail
copy /Y "%PATCH_DIR%line_preprocessing.py" "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" >nul || goto :copy_fail

REM 清 pycache，避免加载旧字节码
echo [清理] __pycache__
for %%D in (
  "%PY_SRC%\mission\__pycache__"
  "%PY_SRC%\tag_pages\__pycache__"
  "%PY_SRC%\ocr\tbpu\parser_tools\__pycache__"
) do if exist %%D rd /s /q %%D 2>nul

echo.
echo === 完成 ===
echo 已部署 5 个宿主补丁。请重新启动 Umi-OCR。
echo 若需回滚，从备份目录拷回:
echo   %BAK%
echo.
pause
exit /b 0

:copy_fail
echo [错误] 复制失败。请确认 Umi-OCR 已退出且对目标目录有写权限。
goto :fail

:fail
echo.
pause
exit /b 1
