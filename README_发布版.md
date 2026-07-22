# Local-Ocr · Umi-OCR 新引擎（PP-OCRv6）发布包

**当前版本：1.1**（与源码仓 `VERSION` / 标签 `v1.1` 对齐）

把 Umi-OCR v2.1.5 内置的 PP-OCRv3 引擎，升级为官方最新 **PaddleOCR 3.x** 路线
（推荐 **PP-OCRv6 medium** + **ONNX Runtime**，可在设置里切 v5/v4 与 Paddle/MKLDNN）。

> **v1.1**：批量文档空白页崩溃与「停止后无法再启」已修；发布包更名为下方两个 zip。  
> 锁定 **`paddlepaddle==3.2.1`**（3.3.x oneDNN 在 Windows/CPU 下易崩，勿擅自升级）。  
> 公开源码：<https://github.com/chapterv/umi-paddle-neoengine>

---

## 版本修订记录

### v1.1（当前）

| 项 | 说明 |
|----|------|
| **批量文档** | 空白页 / 脏空 text 防 `median([])`；单页 OCR 超时后继续 |
| **任务恢复** | Mission `forceRecover`；停止时杀 OCR 引擎，可再次提交任务 |
| **发布包** | 仅保留两包（见下），输出目录 `umi-paddle-neoengine-release/` |

### v1.0 及更早

- 默认 **ONNX Runtime CPU**；可选 ONNX CUDA GPU、Paddle+MKLDNN  
- 部署 `setup.bat`、中文路径、模型自包含 `paddlex/` 等  

---

## 三种推理后端（同一插件文件夹）

插件只有 `Umi-OCR/UmiOCR-data/plugins/win_x64_PaddleOCR_Py/` **一个文件夹**；
三种后端是同一个 `engine.py` 的模式，靠 GUI「推理引擎」切换：

| 路线 | 引擎 | 说明 |
|------|------|------|
| **默认 · ONNX CPU** | `--engine onnxruntime` | 开箱稳，中文路径更不易踩坑 |
| **GPU · ONNX CUDA** | `--engine onnxruntime-gpu` | 优先 CUDA；不可用时回退 CPU |
| **可选 · Paddle MKLDNN** | `--engine paddle` | 须 `paddlepaddle==3.2.1` |

---

## 当前发布包（v1.1）

输出目录：同级 **`umi-paddle-neoengine-release/`**

| 包 | 引擎默认 | 包含 | 解压后 | 适合 |
|----|------|------|--------|------|
| **umi-paddle-neoengine-deploy.zip** | ONNX CPU | 源码 + setup.bat（**不含** venv / 模型） | 双击 `setup.bat` → 打开 Umi-OCR | 有网、最小包 |
| **umi-paddle-neoengine-ONNX-V6-CPU.zip** | ONNX CPU | 含精简 `.venv` + V6 ONNX 等模型 | 直接双击 `Umi-OCR\Umi-OCR.exe` | 懒人 / 离线 |

> 历史包名（`Local-Ocr_*_简洁版/懒人版` 等）已废弃，请使用上表两个文件名。  
> 懒人包模型缓存在插件内 `paddlex/`，自包含。

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
