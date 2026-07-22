# Local-Ocr · Umi-OCR 新引擎（PP-OCRv6 / v4）发布包

把 Umi-OCR v2.1.5 内置的 PP-OCRv3 引擎，升级为官方最新 **PaddleOCR 3.x** 路线
（默认 **PP-OCRv4 mobile**，可在设置里切 **PP-OCRv6 medium**，质量更高）。

> 里程碑：`v1.0-speed-fixed` —— 速度根因已修复（细图不再被放大 40+ 倍）。
> 当前锁定 **`paddlepaddle==3.2.1`**：该版本已修复 oneDNN 的 PIR 崩溃，**MKLDNN 默认开启、稳定加速**。
> ⚠️ 切勿升级到 3.3.x —— 其 oneDNN（MKLDNN）在 Windows/CPU 下推理必崩
> （`ConvertPirAttribute2RuntimeAttribute`）；3.3.x 下只能用下方的「路线二 ONNX」旁路。

---

## 两种推理引擎（同一套代码 · 两个发布包）

插件只有 `Umi-OCR/UmiOCR-data/plugins/win_x64_PaddleOCR_Py/` **一个文件夹**，
MKLDNN 与 ONNX 是同一个 `engine.py` 的**两种后端模式**，靠 GUI「推理引擎」下拉切换：

| 路线 | 引擎 | 说明 |
|------|------|------|
| **路线一 · 纯 MKLDNN（默认）** | `--engine paddle` | Paddle 原生后端 + Intel oneDNN（MKLDNN）CPU 加速。`paddlepaddle==3.2.1` 下稳定、最快。**这是正解 / 完整版。** |
| **路线二 · ONNX Runtime（CPU 旁路）** | `--engine onnxruntime` | 完全绕开 oneDNN，用 ONNX Runtime 的 `CPUExecutionProvider` 推理。用于 3.3.x 下 MKLDNN 崩溃时的兜底 / 对照。两者纯 CPU、同图速度同量级。 |

> **两个发布包共用这一套代码**，区别只在 GUI 里「推理引擎」的**默认值**不同：
> - **MKLDNN 版（双 CPU）**：默认 paddle（MKLDNN），ONNX 仍可在下拉里切换。
> - **ONNX 版（单独 ONNX）**：默认 onnxruntime，直接走 ONNX 旁路。

---

## 本次四个发布包

每个引擎版本都提供 **简洁版**（不含环境/模型，需联网一键部署）与 **懒人版**（含完整 .venv + 模型，解压即用）：

| 包 | 引擎默认 | 包含 | 解压后 | 适合 |
|----|------|------|--------|------|
| **Local-Ocr_MKLDNN_简洁版.zip** | Paddle(MKLDNN) | 源码 + setup.bat（**不含** venv / 模型） | 双击 `setup.bat` 建环境 + 下载模型 → 打开 Umi-OCR | 有网、想拿最小包 |
| **Local-Ocr_MKLDNN_懒人版.zip** | Paddle(MKLDNN) | 全部（含 `.venv/` + `paddlex/` 模型） | 直接双击 `Umi-OCR\Umi-OCR.exe` | 怕麻烦 / 离线机器 |
| **Local-Ocr_ONNX_简洁版.zip** | ONNX Runtime | 源码 + setup.bat（**不含** venv / 模型） | 双击 `setup.bat` 建环境 + 下载模型 → 打开 Umi-OCR | 有网、想走 ONNX 旁路 |
| **Local-Ocr_ONNX_懒人版.zip** | ONNX Runtime | 全部（含 `.venv/` + `paddlex/` 模型） | 直接双击 `Umi-OCR\Umi-OCR.exe` | 怕麻烦 / 离线 / 3.3.x 环境 |

> 原 `Local-Ocr_懒人版.zip` 已重命名为 **`Local-Ocr_MKLDNN_懒人版.zip`** 保留（即双 CPU 完整版）。
> 两个懒人版都**已把模型缓存重定向到插件自己的 `paddlex/` 目录**，故模型自带、行为一致。

---

## `paddlex/` 是干什么的（模型自包含）

`paddlex/` 是本插件**真实 OCR 模型权重的存放目录**（PP-OCRv4 / v5 / v6 的检测、识别、方向分类模型）。

- PaddleOCR 默认把模型下载到系统全局 `~/.paddlex`，那样「模型在别处、插件不自带」。
- 本插件在 `engine.py` 顶部 `import paddleocr` **之前**设置
  `os.environ["PADDLE_PDX_CACHE_HOME"] = <插件目录>/paddlex`，
  **把模型缓存位置重定向到插件自己脚下**。
- 效果：① **懒人版**直接把模型预置进 `paddlex/`，解压即用、离线可用；
  ② **简洁版**首次识别某语言时，paddle 自动把模型下载并缓存到这里，无需手动搬运。
- ⚠️ 仓库根目录那个 `paddlex/`（约 171M）是早期遗留的**废弃缓存**，发布包已排除；
  真正随包的是插件目录内的 `paddlex/`（约 549M）。

---

## 简洁版 用法（两个引擎版通用）

1. 解压 `Local-Ocr_xxx_简洁版.zip`
2. 双击根目录 **`setup.bat`**（两段式）
   - 第 1 步：选模型范围（完整 / 最小可用 / 多语言）
   - 第 2 步：选推理后端 —— `[1]` 纯 GPU（onnxruntime-gpu）/ `[2]` 纯 CPU（默认）/ `[3]` 全安装
   - 选 GPU 时第 3 步：选 CUDA 版本 —— `[1]` 1.27.0 + CUDA13 / `[2]` 1.26.0 + CUDA12.9（默认，RTX 30/40 系）
   - 自动建 `.venv_gpu` 虚拟环境并 `pip install paddlepaddle==3.2.1 paddleocr==3.7.0 onnxruntime-gpu[cuda,cudnn]==1.26.0`
   - 自动预下载所选模型（约 1 分钟，需联网）
   - 全程约 2~3 分钟
   - ⚠️ 选中 GPU 但本机 CUDA 不可用，引擎会明确标 `[⚠回退CPU]` 并打 WARN，功能仍正常（仅无 GPU 提速）
3. 双击 `Umi-OCR\Umi-OCR.exe` 打开主程序
4. 「全局设置 → 文字识别」里选 **PaddleOCR·PP-OCRv6/v4（新引擎）**
5. 拖入图片开始识别（首次切到 v6 会自动下载 v6 模型，仅需一次）

> 前置：本机需有 **Python 3.10 或 3.11**（setup.bat 优先找 3.11）。
> 没装的话先去 https://www.python.org 装一个并勾选 “Add to PATH”。

---

## 懒人版 用法（两个引擎版通用）

1. 解压 `Local-Ocr_xxx_懒人版.zip`（较大，含完整 Python 环境与模型）
2. 直接双击 `Umi-OCR\Umi-OCR.exe`
3. 「全局设置 → 文字识别」里选 **PaddleOCR·PP-OCRv6/v4（新引擎）**
4. 拖入图片即可，**无需联网、无需装任何东西**

> 懒人版的 `.venv` 是 Windows 可移植虚拟环境，换机器解压即用。

---

## 切换推理引擎（两个版都通用）

全局设置 → 文字识别 → 找到本引擎的「推理引擎」选项：

- **Paddle (MKLDNN) 默认**：3.2.1 下稳定、加速。
- **ONNX Runtime（绕过 oneDNN）**：若你手滑升到 `paddlepaddle==3.3.x` 导致 MKLDNN 崩溃，
  切到这里即可正常出结果（绕开 oneDNN）。速度相当，作对照 / 兜底。

---

## 切换模型版本（两个版都通用）

全局设置 → 文字识别 → 找到本引擎的「模型版本」选项：

- **PP-OCRv4 mobile**：默认，速度快。
- **PP-OCRv6 medium**：识别精度更高，但速度较慢。

---

## 本机自测速度（同图 1184×554，warm 单图推理，6 线程 / 边长 1920）

| 后端 | V6 (medium) | V4 (mobile) |
|------|--------------|--------------|
| MKLDNN@3.2.1（路线一·默认） | 7.96s | 7.34s |
| ONNX@3.2.1（路线二） | 7.21s | 5.69s |

> 结论：**ONNX ≤ MKLDNN**（warm 下 V4 反而快 ~22%）。选型看**兼容性**而非速度：
> 3.2.1 的 MKLDNN 稳定、作默认；3.3.x 的 MKLDNN 会崩，只能切 ONNX。
> 大图较慢属正常，可在设置里调大「限制图像边长」或「线程数」。

---

## 目录结构（发布包内）

```
Local-Ocr/
├─ setup.bat                      # 简洁版一键部署
├─ Umi-OCR/
│  ├─ Umi-OCR.exe               # 主程序（官方 v2.1.5）
│  └─ UmiOCR-data/
│     └─ plugins/
│        └─ win_x64_PaddleOCR_Py/   # ← 本项目的全部改动都在这一个文件夹（两引擎共用）
│           ├─ engine.py             # Python 引擎 worker（MKLDNN 修复 + ONNX 旁路）
│           ├─ run.cmd              # 入口：调 .venv/python engine.py
│           ├─ PPOCR_*.py          # Umi-OCR 插件契约
│           ├─ requirements.txt     # paddlepaddle==3.2.1 / paddleocr==3.7.0 / onnxruntime-gpu[cuda,cudnn]==1.26.0（默认 CUDA 12.9）/ 1.27.0（CUDA 13）
│           ├─ models/configs.txt  # 语言下拉框配置（必须保留）
│           ├─ .venv_gpu/        # [仅懒人版] Python 环境（GPU 用 onnxruntime-gpu）
│           └─ paddlex/           # [仅懒人版] 官方模型缓存（真实权重）
└─ README_发布版.md
```

> 想自己改引擎代码？所有改动只在 `plugins/win_x64_PaddleOCR_Py/` 一个目录，
> Umi-OCR 主程序零改动。
