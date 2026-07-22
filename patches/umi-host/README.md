# Umi 宿主补丁（v1.1）

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

1. 校验 5 个补丁文件齐全  
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
| **apply_host_patches.bat** | （部署脚本，不覆盖到 Umi） |

## v1.1 修复摘要

- 空白页 / 脏空 text：`linePreprocessing` 防 `median([])` 崩溃  
- Mission 工人：`msnTask` try/except + `finally` 复位；`forceRecover` + worker epoch  
- 批量文档：OCR 等待约 180s 超时；stop 杀引擎并强制恢复，可再次提交  

## 说明

- 这些文件属于 **Umi 主程序（宿主）**，不是 `win_x64_PaddleOCR_Py` 插件本体。  
- 公开仓以 `patches/umi-host/` 归档；完整 zip 包已直接打进 `py_src`。  
