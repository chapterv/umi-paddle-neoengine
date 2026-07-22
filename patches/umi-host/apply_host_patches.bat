@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo === umi-paddle-neoengine host patch deploy v1.1 ===
echo Patch dir: %~dp0
echo.

set "PATCH_DIR=%~dp0"
set "TARGET=%~1"

REM ---- check patch files ----
set "MISS=0"
if not exist "%PATCH_DIR%mission.py" set "MISS=1"
if not exist "%PATCH_DIR%mission_doc.py" set "MISS=1"
if not exist "%PATCH_DIR%mission_ocr.py" set "MISS=1"
if not exist "%PATCH_DIR%BatchDOC.py" set "MISS=1"
if not exist "%PATCH_DIR%line_preprocessing.py" set "MISS=1"
if "!MISS!"=="1" (
  echo [ERROR] Missing patch py files in this folder.
  goto FAIL
)

REM ---- resolve target ----
if not "%TARGET%"=="" goto RESOLVE

REM auto-detect common paths
if exist "%PATCH_DIR%..\..\..\Local-Ocr\Umi-OCR\UmiOCR-data\py_src\mission\mission.py" (
  set "TARGET=%PATCH_DIR%..\..\..\Local-Ocr\Umi-OCR"
  goto RESOLVE
)
if exist "%PATCH_DIR%..\..\Umi-OCR\UmiOCR-data\py_src\mission\mission.py" (
  set "TARGET=%PATCH_DIR%..\..\Umi-OCR"
  goto RESOLVE
)
if exist "C:\tools\Umi-OCR\UmiOCR-data\py_src\mission\mission.py" (
  set "TARGET=C:\tools\Umi-OCR"
  goto RESOLVE
)
if exist "C:\tools\Umi-Ocr\UmiOCR-data\py_src\mission\mission.py" (
  set "TARGET=C:\tools\Umi-Ocr"
  goto RESOLVE
)

echo [ERROR] Umi-OCR not found. Drag Umi-OCR folder onto this bat, or run:
echo   apply_host_patches.bat "D:\path\to\Umi-OCR"
goto FAIL

:RESOLVE
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET:~0,-1%"
if "%TARGET:~-1%"=="/" set "TARGET=%TARGET:~0,-1%"

set "PY_SRC="
if exist "%TARGET%\UmiOCR-data\py_src\mission\mission.py" set "PY_SRC=%TARGET%\UmiOCR-data\py_src"
if exist "%TARGET%\py_src\mission\mission.py" set "PY_SRC=%TARGET%\py_src"
if exist "%TARGET%\mission\mission.py" set "PY_SRC=%TARGET%"
if exist "%TARGET%\Umi-OCR\UmiOCR-data\py_src\mission\mission.py" set "PY_SRC=%TARGET%\Umi-OCR\UmiOCR-data\py_src"

if not defined PY_SRC (
  echo [ERROR] Cannot resolve py_src under: %TARGET%
  goto FAIL
)

echo Target py_src: %PY_SRC%
echo.

REM ---- backup ----
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "BAK=%PY_SRC%\_patch_backup_%TS%"
mkdir "%BAK%\mission" 2>nul
mkdir "%BAK%\tag_pages" 2>nul
mkdir "%BAK%\parser_tools" 2>nul
echo [backup] %BAK%
copy /Y "%PY_SRC%\mission\mission.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_doc.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_ocr.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\tag_pages\BatchDOC.py" "%BAK%\tag_pages\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" "%BAK%\parser_tools\" >nul 2>&1

REM ---- apply ----
echo [apply] writing patches...
copy /Y "%PATCH_DIR%mission.py" "%PY_SRC%\mission\mission.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%mission_doc.py" "%PY_SRC%\mission\mission_doc.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%mission_ocr.py" "%PY_SRC%\mission\mission_ocr.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%BatchDOC.py" "%PY_SRC%\tag_pages\BatchDOC.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%line_preprocessing.py" "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" >nul
if errorlevel 1 goto COPYFAIL

if exist "%PY_SRC%\mission\__pycache__" rd /s /q "%PY_SRC%\mission\__pycache__" 2>nul
if exist "%PY_SRC%\tag_pages\__pycache__" rd /s /q "%PY_SRC%\tag_pages\__pycache__" 2>nul
if exist "%PY_SRC%\ocr\tbpu\parser_tools\__pycache__" rd /s /q "%PY_SRC%\ocr\tbpu\parser_tools\__pycache__" 2>nul

echo.
echo === DONE ===
echo 5 host patches applied. Restart Umi-OCR.
echo Rollback from: %BAK%
echo.
pause
exit /b 0

:COPYFAIL
echo [ERROR] Copy failed. Exit Umi-OCR fully and check write permission.
goto FAIL

:FAIL
echo.
pause
exit /b 1
