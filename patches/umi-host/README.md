# Umi 宿主补丁（v1.3）

完整发布包（deploy / ONNX-V6-CPU）已内嵌到 Umi `py_src`。  
本目录用于：**官方 Umi 安装** 或 **仅装插件、主程序仍是原版** 时，一键覆盖宿主修复。

## 一键部署（推荐）

双击或命令行运行：

```bat
apply_host_patches.bat
```

或拖入 / 指定 Umi 路径：

```bat
apply_host_patches.bat "D:\Download\umi-cpu\Umi-OCR"
apply_host_patches.bat "C:\tools\Umi-OCR"
```

脚本会：

1. 校验 19 个补丁文件齐全
2. 解析 `Umi-OCR` / `UmiOCR-data` / `py_src` 路径  
3. 备份原文件到 `py_src\_patch_backup_时间戳\`  
4. 覆盖写入并清理相关 `__pycache__`  

**请先完全退出 Umi-OCR（含托盘）再执行。**

## 文件对应

| 本目录文件 | 覆盖到 Umi 路径 |
|------------|-----------------|
| mission.py | `UmiOCR-data/py_src/mission/mission.py` |
| mission_doc.py | `UmiOCR-data/py_src/mission/mission_doc.py` |
| mission_ocr.py | `UmiOCR-data/py_src/mission/mission_ocr.py` |
| BatchDOC.py | `UmiOCR-data/py_src/tag_pages/BatchDOC.py` |
| line_preprocessing.py | `UmiOCR-data/py_src/ocr/tbpu/parser_tools/line_preprocessing.py` |
| output_init.py | `UmiOCR-data/py_src/ocr/output/__init__.py` |
| output_table_csv.py | `UmiOCR-data/py_src/ocr/output/output_table_csv.py` |
| output_tools.py | `UmiOCR-data/py_src/ocr/output/tools.py` |
| output_pdf_layered.py | `UmiOCR-data/py_src/ocr/output/output_pdf_layered.py` |
| output_pdf_one_layer.py | `UmiOCR-data/py_src/ocr/output/output_pdf_one_layer.py` |
| tbpu_init.py | `UmiOCR-data/py_src/ocr/tbpu/__init__.py` |
| parser_table_grid.py | `UmiOCR-data/py_src/ocr/tbpu/parser_table_grid.py` |
| table_grid.py | `UmiOCR-data/py_src/ocr/tbpu/parser_tools/table_grid.py` |
| UtilsConfigDicts.qml | `UmiOCR-data/qt_res/qml/Configs/UtilsConfigDicts.qml` |
| ConfigItemComp.qml | `UmiOCR-data/qt_res/qml/Configs/ConfigItemComp.qml` |
| Configs.qml | `UmiOCR-data/qt_res/qml/Configs/Configs.qml` |
| BatchDOCConfigs.qml | `UmiOCR-data/qt_res/qml/TabPages/BatchDOC/BatchDOCConfigs.qml` |
| BatchOCRConfigs.qml | `UmiOCR-data/qt_res/qml/TabPages/BatchOCR/BatchOCRConfigs.qml` |
| ResultsTableView.qml | `UmiOCR-data/qt_res/qml/Widgets/ResultLayout/ResultsTableView.qml` |
| **apply_host_patches.bat** | （部署脚本，不覆盖到 Umi） |

## v1.3 修复摘要

- 双层 PDF 自动识别横排与竖排文字框，竖排字符保持直立并对齐原图。
- 竖排段落复制按 OCR 阅读顺序输出：栏内从上到下、栏间从右到左，不再被
  Chrome / Edge 按页面横坐标重排。
- 单层 `text.pdf` 新增自动、统一横排、统一竖排三种排版方向。

## v1.2 修复摘要

- P0 `表格-几何网格` 排版解析
- `table.csv 结构表格(Excel)` 输出与 UTF-8 BOM
- tableCsv 在普通排版前自动从原始 OCR 坐标建表
- 结果区 TSV 表格预览

## v1.1 修复摘要

- 空白页 / 脏空 text：`linePreprocessing` 防 `median([])` 崩溃  
- Mission 工人：`msnTask` try/except + `finally` 复位；`forceRecover` + worker epoch  
- 批量文档：OCR 等待约 180s 超时；stop 杀引擎并强制恢复，可再次提交  

## 说明

- 这些文件属于 **Umi 主程序（宿主）**，不是 `win_x64_PaddleOCR_Py` 插件本体。  
- 公开仓以 `patches/umi-host/` 归档；完整 zip 包已直接打进 `py_src`。  
