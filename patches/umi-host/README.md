# Umi 宿主补丁（v1.1）

完整发布包（deploy / ONNX-V6-CPU）已内嵌到 Umi `py_src`。
本目录便于只更新插件仓时对照 / 手工覆盖。

| 文件 | 覆盖到 Umi 路径 |
|------|----------------|
| mission.py | UmiOCR-data/py_src/mission/mission.py |
| mission_doc.py | UmiOCR-data/py_src/mission/mission_doc.py |
| mission_ocr.py | UmiOCR-data/py_src/mission/mission_ocr.py |
| BatchDOC.py | UmiOCR-data/py_src/tag_pages/BatchDOC.py |
| line_preprocessing.py | UmiOCR-data/py_src/ocr/tbpu/parser_tools/line_preprocessing.py |

## v1.1 修复摘要

- 空白页 / 脏空 text：`linePreprocessing` 防 `median([])` 崩溃
- Mission 工人：`msnTask` try/except + `finally` 复位；`forceRecover` + worker epoch
- 批量文档：OCR 等待 180s 超时；stop 杀引擎并强制恢复，可再次提交
