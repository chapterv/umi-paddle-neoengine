# Local-Ocr-Plugin · PaddleOCR·PP-OCRv6 插件（独立仓库）

本仓库**只包含插件源码**，用于分享/接入 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR)。
完整发布包（含主程序 + 一键部署脚本）见 `Local-Ocr_简洁版.zip` / `Local-Ocr_懒人版.zip`。

## 目录结构
```
Local-Ocr-Plugin/
└── win_x64_PaddleOCR_Py/      ← 这就是插件本体（丢进 Umi-OCR 的 plugins 目录）
    ├── __init__.py
    ├── engine.py              ← 引擎入口（含 Path-1 修复：推理期 stdout→stderr 重定向，避免 oneDNN 诊断信息污染 JSON 协议导致 904）
    ├── PPOCR_umi.py           ← 对接 Umi-OCR 的 Api 类
    ├── PPOCR_api.py           ← 子进程 worker
    ├── PPOCR_config.py        ← 引擎选项（ocr 版本 / MKLDNN 开关）
    ├── run.cmd
    ├── requirements.txt       ← paddlepaddle==3.2.1 + paddleocr==3.7.0（⚠️ 必须 3.2.1）
    ├── i18n.csv
    ├── models/                ← 语言下拉框配置（config_*.txt + configs.txt）
    └── README.md              ← 插件内说明
```

## 安装到 Umi-OCR
1. 把 `win_x64_PaddleOCR_Py/` 整个文件夹复制到 Umi-OCR 的
   `UmiOCR-data/plugins/` 目录下。
2. 进入 `win_x64_PaddleOCR_Py/`，用 **Python 3.11** 建虚拟环境并装依赖：
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. 打开 Umi-OCR → 全局设置 → 文字识别，选择
   **“PaddleOCR·PP-OCRv6（新引擎）”**，拖图识别即可。
   （首次切到 v6 会自动联网下载模型。）

## 重要版本约束
- **paddlepaddle 必须 == 3.2.1**。3.3.x 的 PIR+oneDNN 兼容性 bug 会让 MKLDNN
  推理崩溃（`ConvertPirAttribute2RuntimeAttribute` ArrayAttribute），详见
  PaddlePaddle#77340 / PaddleOCR#17869。本项目已通过 Path-1 修复 + 固定 3.2.1 解决。
- MKLDNN 默认开启；若某天 MKL 再出同类 bug，可在 `PPOCR_config.py` 关闭，
  或在 `engine.py` 透传 `engine='onnxruntime'`（已接好 `--engine` 参数，作为兜底）。

## 许可
插件本体遵循 Umi-OCR 相关许可；修改与打包按各自仓库约定。
