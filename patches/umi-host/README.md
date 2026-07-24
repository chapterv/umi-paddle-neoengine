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

1. 校验 26 个宿主/插件补丁文件齐全
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
| BatchOCR.py | `UmiOCR-data/py_src/tag_pages/BatchOCR.py` |
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
| PPOCR_umi.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/PPOCR_umi.py` |
| PPOCR_config.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/PPOCR_config.py` |
| engine.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/engine.py` |
| model_sources.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/model_sources.py` |
| table_structure.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/table_structure.py` |
| punctuation_recovery.py | `UmiOCR-data/plugins/win_x64_PaddleOCR_Py/punctuation_recovery.py` |
| **apply_host_patches.bat** | （部署脚本，不覆盖到 Umi） |

## v1.3 修复摘要

- 双层 PDF 每个竖排 OCR 栏采用标准嵌入 CJK 连续 run，保障局部高亮与单栏连续复制。
  Chrome/Edge 跨栏仍按几何左→右并可能换行；需严格语序请导出 TXT 或单层统一横排。
- 单层 `text.pdf` 新增自动、统一横排、统一竖排三种排版方向。
- P1 表格模型按表格 CSV 的明确请求触发；普通 OCR、截图与预览不会加载它。
- 可选 JSONL 追踪将原始 OCR、预览和文档导出以同一请求 ID 记录，便于核对标点。
- C3 默认内部启用：仅依据竖排块原图中的高置信物理形状恢复 `。`、`，`、`：`；
  分别要求环形连通域、孤立短笔画或两个小而对齐的墨点，并同时检查尺寸、视觉
  cell、低墨量及字符/词框插入位置；记录 `image_connected_component`、bbox、
  插入位置和置信度，绝不由换行或语义猜测。紧急回退可在启动前设置
  `UMI_OCR_VERTICAL_PUNCTUATION_RECOVERY=0`。

### 标点调试追踪（默认关闭）

在 PaddleOCR 插件设置的“OCR 调试追踪 JSONL（可选）”填写 JSONL 文件路径后，
批量文档会为每页记录 `raw_ocr`、`preview`、`document_export` 三个阶段及同一
`request_id`。留空时不写文件；该功能只保存证据，不会根据换行补标点。

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
