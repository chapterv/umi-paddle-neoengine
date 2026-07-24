@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo === umi-paddle-neoengine host patch deploy v1.3 ===
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
if not exist "%PATCH_DIR%BatchOCR.py" set "MISS=1"
if not exist "%PATCH_DIR%line_preprocessing.py" set "MISS=1"
if not exist "%PATCH_DIR%output_init.py" set "MISS=1"
if not exist "%PATCH_DIR%output_table_csv.py" set "MISS=1"
if not exist "%PATCH_DIR%output_tools.py" set "MISS=1"
if not exist "%PATCH_DIR%output_pdf_layered.py" set "MISS=1"
if not exist "%PATCH_DIR%output_pdf_one_layer.py" set "MISS=1"
if not exist "%PATCH_DIR%tbpu_init.py" set "MISS=1"
if not exist "%PATCH_DIR%parser_table_grid.py" set "MISS=1"
if not exist "%PATCH_DIR%table_grid.py" set "MISS=1"
if not exist "%PATCH_DIR%UtilsConfigDicts.qml" set "MISS=1"
if not exist "%PATCH_DIR%ConfigItemComp.qml" set "MISS=1"
if not exist "%PATCH_DIR%Configs.qml" set "MISS=1"
if not exist "%PATCH_DIR%BatchDOCConfigs.qml" set "MISS=1"
if not exist "%PATCH_DIR%BatchOCRConfigs.qml" set "MISS=1"
if not exist "%PATCH_DIR%ResultsTableView.qml" set "MISS=1"
if not exist "%PATCH_DIR%PPOCR_umi.py" set "MISS=1"
if not exist "%PATCH_DIR%PPOCR_config.py" set "MISS=1"
if not exist "%PATCH_DIR%engine.py" set "MISS=1"
if not exist "%PATCH_DIR%model_sources.py" set "MISS=1"
if not exist "%PATCH_DIR%table_structure.py" set "MISS=1"
if not exist "%PATCH_DIR%punctuation_recovery.py" set "MISS=1"
if "!MISS!"=="1" (
  echo [ERROR] Missing host patch files in this folder.
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

for %%I in ("%PY_SRC%\..") do set "DATA_ROOT=%%~fI"
if not exist "%DATA_ROOT%\qt_res\qml\Configs\UtilsConfigDicts.qml" (
  echo [ERROR] Cannot resolve qt_res\qml under: %DATA_ROOT%
  goto FAIL
)

echo Target py_src: %PY_SRC%
echo Target UmiOCR-data: %DATA_ROOT%
echo.

REM ---- backup ----
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "BAK=%PY_SRC%\_patch_backup_%TS%"
mkdir "%BAK%\mission" 2>nul
mkdir "%BAK%\tag_pages" 2>nul
mkdir "%BAK%\parser_tools" 2>nul
mkdir "%BAK%\ocr\output" 2>nul
mkdir "%BAK%\ocr\tbpu\parser_tools" 2>nul
mkdir "%BAK%\qt_res\qml\Configs" 2>nul
mkdir "%BAK%\qt_res\qml\TabPages\BatchDOC" 2>nul
mkdir "%BAK%\qt_res\qml\TabPages\BatchOCR" 2>nul
mkdir "%BAK%\qt_res\qml\Widgets\ResultLayout" 2>nul
mkdir "%BAK%\plugins\win_x64_PaddleOCR_Py" 2>nul
echo [backup] %BAK%
copy /Y "%PY_SRC%\mission\mission.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_doc.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\mission\mission_ocr.py" "%BAK%\mission\" >nul 2>&1
copy /Y "%PY_SRC%\tag_pages\BatchDOC.py" "%BAK%\tag_pages\" >nul 2>&1
copy /Y "%PY_SRC%\tag_pages\BatchOCR.py" "%BAK%\tag_pages\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" "%BAK%\parser_tools\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\output\__init__.py" "%BAK%\ocr\output\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\output\output_table_csv.py" "%BAK%\ocr\output\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\output\tools.py" "%BAK%\ocr\output\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\output\output_pdf_layered.py" "%BAK%\ocr\output\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\output\output_pdf_one_layer.py" "%BAK%\ocr\output\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\__init__.py" "%BAK%\ocr\tbpu\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\parser_table_grid.py" "%BAK%\ocr\tbpu\" >nul 2>&1
copy /Y "%PY_SRC%\ocr\tbpu\parser_tools\table_grid.py" "%BAK%\ocr\tbpu\parser_tools\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\Configs\UtilsConfigDicts.qml" "%BAK%\qt_res\qml\Configs\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\Configs\ConfigItemComp.qml" "%BAK%\qt_res\qml\Configs\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\Configs\Configs.qml" "%BAK%\qt_res\qml\Configs\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\TabPages\BatchDOC\BatchDOCConfigs.qml" "%BAK%\qt_res\qml\TabPages\BatchDOC\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\TabPages\BatchOCR\BatchOCRConfigs.qml" "%BAK%\qt_res\qml\TabPages\BatchOCR\" >nul 2>&1
copy /Y "%DATA_ROOT%\qt_res\qml\Widgets\ResultLayout\ResultsTableView.qml" "%BAK%\qt_res\qml\Widgets\ResultLayout\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\PPOCR_umi.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\PPOCR_config.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\engine.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\model_sources.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\table_structure.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1
copy /Y "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\punctuation_recovery.py" "%BAK%\plugins\win_x64_PaddleOCR_Py\" >nul 2>&1

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
copy /Y "%PATCH_DIR%BatchOCR.py" "%PY_SRC%\tag_pages\BatchOCR.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%line_preprocessing.py" "%PY_SRC%\ocr\tbpu\parser_tools\line_preprocessing.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%output_init.py" "%PY_SRC%\ocr\output\__init__.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%output_table_csv.py" "%PY_SRC%\ocr\output\output_table_csv.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%output_tools.py" "%PY_SRC%\ocr\output\tools.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%output_pdf_layered.py" "%PY_SRC%\ocr\output\output_pdf_layered.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%output_pdf_one_layer.py" "%PY_SRC%\ocr\output\output_pdf_one_layer.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%tbpu_init.py" "%PY_SRC%\ocr\tbpu\__init__.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%parser_table_grid.py" "%PY_SRC%\ocr\tbpu\parser_table_grid.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%table_grid.py" "%PY_SRC%\ocr\tbpu\parser_tools\table_grid.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%UtilsConfigDicts.qml" "%DATA_ROOT%\qt_res\qml\Configs\UtilsConfigDicts.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%ConfigItemComp.qml" "%DATA_ROOT%\qt_res\qml\Configs\ConfigItemComp.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%Configs.qml" "%DATA_ROOT%\qt_res\qml\Configs\Configs.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%BatchDOCConfigs.qml" "%DATA_ROOT%\qt_res\qml\TabPages\BatchDOC\BatchDOCConfigs.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%BatchOCRConfigs.qml" "%DATA_ROOT%\qt_res\qml\TabPages\BatchOCR\BatchOCRConfigs.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%ResultsTableView.qml" "%DATA_ROOT%\qt_res\qml\Widgets\ResultLayout\ResultsTableView.qml" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%PPOCR_umi.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\PPOCR_umi.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%PPOCR_config.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\PPOCR_config.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%engine.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\engine.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%model_sources.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\model_sources.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%table_structure.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\table_structure.py" >nul
if errorlevel 1 goto COPYFAIL
copy /Y "%PATCH_DIR%punctuation_recovery.py" "%DATA_ROOT%\plugins\win_x64_PaddleOCR_Py\punctuation_recovery.py" >nul
if errorlevel 1 goto COPYFAIL

if exist "%PY_SRC%\mission\__pycache__" rd /s /q "%PY_SRC%\mission\__pycache__" 2>nul
if exist "%PY_SRC%\tag_pages\__pycache__" rd /s /q "%PY_SRC%\tag_pages\__pycache__" 2>nul
if exist "%PY_SRC%\ocr\output\__pycache__" rd /s /q "%PY_SRC%\ocr\output\__pycache__" 2>nul
if exist "%PY_SRC%\ocr\tbpu\__pycache__" rd /s /q "%PY_SRC%\ocr\tbpu\__pycache__" 2>nul
if exist "%PY_SRC%\ocr\tbpu\parser_tools\__pycache__" rd /s /q "%PY_SRC%\ocr\tbpu\parser_tools\__pycache__" 2>nul

echo.
echo === DONE ===
echo 26 host/plugin patches applied. Restart Umi-OCR.
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
