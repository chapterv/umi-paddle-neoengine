# Local-Ocr-Plugin · PaddleOCR·PP-OCRv6 插件（独立仓库）

本仓库**只包含插件源码**，用于分享 / 接入 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR)。
完整发布包（含主程序 + 一段式部署脚本）见仓库同级目录 `Local-Ocr_发布包/`。

## 这是什么

一个 Umi-OCR 插件 `win_x64_PaddleOCR_Py/`，把官方最新 **PaddleOCR 3.x（PP-OCRv6 / v5 / v4）**
接入 Umi-OCR，**Umi-OCR 主程序零改动**。

**单一插件、单一 `engine.py`**，同时支持三种推理后端，靠 GUI「推理引擎」下拉框 /
`--engine` 参数切换，**不拆目录、不拆插件**：

| 后端 | 说明 |
|------|------|
| **ONNX Runtime · CUDA GPU** | `--engine onnxruntime-gpu`。优先 `CUDAExecutionProvider`，**不可用时自动回退 CPU**（CUDA 缺失 / cuDNN 未装 / 某 op 不支持均安全降级，功能正常只是无 GPU 提速）。由 `requirements.txt` 的 `onnxruntime-gpu[cuda,cudnn]` 提供，CUDA 12.x DLL 随包装进 `.venv_gpu`，**无需系统装 CUDA Toolkit**。 |
| **ONNX Runtime · CPU** | `--engine onnxruntime`。绕开 oneDNN，用 `CPUExecutionProvider` 推理；用作兜底 / 对照。 |
| **Paddle (MKLDNN) · CPU** | `--engine paddle`。Paddle 原生后端 + Intel oneDNN（MKLDNN）CPU 加速。锁定 `paddlepaddle==3.2.1`（该版已修复 oneDNN 的 PIR 崩溃）。 |

> **选型看兼容性而非速度**：3.2.1 的 MKLDNN 稳定、作默认；3.3.x 的 MKLDNN 会崩，
> 只能切 ONNX。两者纯 CPU 时速度同量级。GPU（ONNX-CUDA）则显著快于纯 CPU。

## 目录结构

```
Local-Ocr-Plugin/
└── win_x64_PaddleOCR_Py/      ← 这就是插件本体（丢进 Umi-OCR 的 plugins 目录）
    ├── __init__.py
    ├── engine.py              ← 引擎入口（含修复：推理期 stdout→stderr 重定向，避免 oneDNN 诊断信息污染 JSON 协议导致 904；CUDA 不可用自动回退 CPU 并在日志标 [⚠回退CPU]）
    ├── PPOCR_umi.py           ← 对接 Umi-OCR 的 Api 类
    ├── PPOCR_api.py           ← 子进程 worker
    ├── PPOCR_config.py        ← 引擎选项（推理引擎 paddle/onnxruntime 可切换、语言、方向分类、边长限制等）
    ├── run.cmd                ← 入口：调本目录 .venv_gpu 的 python 跑 engine.py
    ├── requirements.txt       ← paddlepaddle==3.2.1 + paddleocr==3.7.0 + onnxruntime-gpu[cuda,cudnn]==1.26.0（默认 CUDA 12.9）/ 1.27.0（CUDA 13）
    ├── i18n.csv
    ├── models/                ← 语言下拉框配置（config_*.txt + configs.txt，必须保留）
    └── README.md             ← 插件内说明（后端细节 / paddlex 用途 / 速度实测 / 部署）
```

> `.venv_gpu/`、`paddlex/`、`__pycache__/`、`*.log` 均不进本仓库（见 `.gitignore`）。

## 安装到 Umi-OCR

1. 把 `win_x64_PaddleOCR_Py/` 整个文件夹复制到 Umi-OCR 的
   `UmiOCR-data/plugins/` 目录下。
2. **首次使用必须先部署依赖**（建 Python 环境并装 paddle + paddleocr + onnxruntime-gpu）。
   最简单：直接用 `Local-Ocr_发布包/` 里的 `setup.bat`（根目录，两段式：
   第 1 步选模型范围 → 第 2 步选推理后端 GPU/CPU → 选 GPU 时第 3 步追问 CUDA 版本）。
   它会自动建 `.venv_gpu`、装依赖、验证 providers。
   手动等价做法（用 Windows 原生路径，避免把 `/c/...` 错拼成 `C:\c\...`）：
   ```bat
   cd Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py
   uv venv --python 3.11 .venv_gpu
   uv pip install --python .venv_gpu\Scripts\python.exe -r requirements.txt
   ```
3. 打开 Umi-OCR → 全局设置 → 文字识别，选择
   **“PaddleOCR·PP-OCRv6（新引擎）”**，拖图识别即可。
   （首次识别某语言会自动把模型下载并缓存到插件自己的 `paddlex/` 目录。）

## 验证 / 排查要点

- **后端标签**：识别结果列表的「耗时」右侧会显示后端，如 `gpu(cuda) v6`（GPU 命中）/
  `cpu v6`（回退 CPU 或纯 CPU 后端），便于确认实际走了哪条路径。
- **GPU 没生效？** 日志（`engine_active.log` 与 stderr）会明确标 `[⚠回退CPU]` 并打 WARN；
  重跑 `setup.bat` 选 GPU 项（[1]/[2]）装 `onnxruntime-gpu` 即可。
- **paddlepaddle 必须 == 3.2.1**：3.3.x 的 oneDNN（MKLDNN）在 Windows/CPU 下推理必崩
  （`ConvertPirAttribute2RuntimeAttribute`）；本项目已固定 3.2.1 + 路径 1 修复解决。
- **语言 → 模型代自动回退**：PP-OCRv6 覆盖 简/繁/英/日 + 拉丁语系；
  韩文、俄文 v6 不支持，自动回退到 PP-OCRv5（识别正常）。
- **俄文键是 `ru`**（paddleocr 3.x 用户态码），本插件已做 `cyrillic→ru` 映射。

## 许可

插件本体遵循 Umi-OCR 相关许可；修改与打包按各自仓库约定。
