# win_x64_PaddleOCR_Py —— Umi-OCR 的 PaddleOCR（Python）引擎插件

基于 **PaddleOCR 3.x（pipeline 架构，PP-OCRv6/v5）** 的本地 OCR 引擎，
以 Umi-OCR 插件形式提供，与官方 `hiroi-sora/PaddleOCR-json` 管道协议完全兼容，
**Umi-OCR 主程序零改动**。

> 路线：自写 Python 插件（Route B），引擎子进程直接调用官方 `paddleocr.PaddleOCR()`
> 推理，复用 Umi-OCR 自带的 `PPOCR_umi` / `PPOCR_api` JSON 管道。
> 旧引擎 `win7_x64_PaddleOCR-json` 保留作为回退，本插件是「新增可选引擎」。

## 三种后端（同一文件夹 · 单一插件）
本插件**只有 `win_x64_PaddleOCR_Py` 这一个文件夹、一个 `engine.py`**；**paddle(MKLDNN) / ONNX-CPU / ONNX-CUDA
是同一个 `engine.py` 的三种推理后端**，靠 `--engine` 参数（GUI「推理引擎」下拉框）切换。
**不拆目录、不拆插件**，一份代码覆盖全部后端：

- **路线一 · 纯 MKLDNN（默认 · 本项目正解）**：`--engine paddle`（或默认/空）。Paddle 原生后端 + Intel oneDNN（MKLDNN）CPU 加速。
  锁定 `paddlepaddle==3.2.1`（该版已修复 oneDNN 的 PIR 崩溃）；**MKLDNN 默认开启、稳定加速**。
  → 这是「完整版 / 双 CPU 版」的默认引擎。
- **路线二 · ONNX Runtime（CPU 旁路）**：`--engine onnxruntime`。完全绕开 oneDNN，用 ONNX Runtime 的 `CPUExecutionProvider` 推理。
  用途：① `paddlepaddle==3.3.x`（最新）下 MKLDNN 会崩溃，此时只能切 ONNX 才能正常出结果；
  ② 作为对照 / 兜底。两者纯 CPU、同图速度同量级（见下方实测表），选型看**兼容性**而非速度。
- **路线三 · ONNX Runtime CUDA GPU**：`--engine onnxruntime-gpu`。优先 `CUDAExecutionProvider`，**不可用时自动回退 CPU**
  （CUDA 缺失 / cuDNN 未装 / 某 op 不支持均安全降级，功能正常只是无 GPU 提速）。
  由 `requirements.txt` 中的 `onnxruntime-gpu[cuda,cudnn]` 提供，CUDA 12.x DLL 随包装进 `.venv`，**无需系统装 CUDA Toolkit**；
  驱动最高仅到 CUDA 12.9 的环境（如 RTX 3070 Ti + 驱动 576.57）用 1.26.0，支持 CUDA 13（R580+）的可换 1.27.0。

## 文件说明
| 文件 | 作用 |
|---|---|
| `run.cmd` | 引擎入口。Umi-OCR 调它 → 用本目录 `.venv_gpu` 的 python 跑 `engine.py` |
| `engine.py` | **worker**：解析参数 → 建 `PaddleOCR` → `OCR init completed.` 握手 → 逐行读 JSON 识图 |
| `PPOCR_api.py` | 管道客户端（从 Umi-OCR 自带 Python 运行，不依赖 paddle） |
| `PPOCR_umi.py` | Umi-OCR 插件接口 `class Api`（标准契约） |
| `PPOCR_config.py` / `i18n.csv` | 引擎设置面板（语言下拉框、方向分类、边长限制等） |
| `models/configs.txt` + `config_*.txt` | **仅驱动语言下拉框**，引擎不读其内容；真实模型由 paddle 自动下载 |
| `requirements.txt` | `paddlepaddle==3.2.1` + `paddleocr==3.7.0` + `onnxruntime-gpu[cuda,cudnn]==1.26.0`（默认 CUDA 12.9）/ `1.27.0`（CUDA 13） |
| `.venv/` | 隔离 Python 环境（含 paddle，已装；**不进 git**） |
| `paddlex/` | **真实 OCR 模型权重目录**（PP-OCRv4 / v5 / v6 的检测+识别+方向分类模型） |

## `paddlex/` 是干什么的（模型自包含）
`paddlex/` 是本插件**真实 OCR 模型权重的存放目录**（PP-OCRv4 / v5 / v6 的检测、文字识别、方向分类模型）。
为什么需要它、它解决什么问题：

- PaddleOCR 默认把模型下载到系统全局 `~/.paddlex/official_models`，那样「模型在别处、插件不自带」，换机器就得重下。
- 本插件在 `engine.py` 顶部 `import paddleocr` **之前**设置
  `os.environ["PADDLE_PDX_CACHE_HOME"] = <插件目录>/paddlex`，
  **把模型缓存位置重定向到插件自己脚下**。
- 效果：
  ① **懒人版**直接把模型预置进 `paddlex/`，**解压即用、离线可用**；
  ② **简洁版**首次识别某语言时，paddle 自动把模型下载并缓存到这里，**无需你手动搬运**。
- ⚠️ 根目录那个 `paddlex/`（约 171M）是早期遗留的**废弃缓存**，发布包已排除；
  真正随包的是插件目录内的 `paddlex/`（约 549M）。

## 使用方法（Umi-OCR GUI）
1. 用 **Umi-OCR v2.1.5**（或兼容版本）打开本项目 `Umi-OCR/` 目录的程序。
2. 全局设置 → 「引擎」下拉框，选择本插件（名为 **PaddleOCR（本地）** / 对应 `win_x64_PaddleOCR_Py`）。
3. 「文字识别 → 语言/模型库」选择所需语言：
   - 简体中文 / English / 繁體中文 / 日本語 / 한국어 / Русский（俄文）
4. 模型位置由 `engine.py` 通过环境变量 `PADDLE_PDX_CACHE_HOME` **重定向到插件自己的 `paddlex/` 目录**
   （`Umi-OCR/UmiOCR-data/plugins/win_x64_PaddleOCR_Py/paddlex/`），与系统全局 `~/.paddlex` 无关。
   懒人版已把模型预置到此目录；简洁版首次识别某语言时，paddle 会**自动下载并缓存**到这里
   （约几十~上百 MB，只需一次，无需手动搬运）。
5. 拖入图片或粘贴截图即可识别。

## 已知约束（务必阅读）
- **MKLDNN 默认开启（已修复）**：paddle 3.3.x 的 oneDNN 在 Windows/CPU 下推理期崩溃
  （`ConvertPirAttribute2RuntimeAttribute ... onednn`），故本插件**锁定 paddlepaddle==3.2.1**
  （该版本已修复此 PIR bug）。`enable_mkldnn` 默认 True，GUI 勾选即生效，正常加速。
  ⚠️ 切勿升级到 3.3.x，否则 MKLDNN 会崩溃。
- **904 修复**：paddle/oneDNN 会在 stdout 打印 `[ReduceMeanCheckIfOneDNNSupport]` 诊断信息，
  污染 Umi-OCR 的 JSON 结果行导致「904 反序列化失败」。引擎已在每次推理期间把 stdout
  临时重定向到 stderr（`os.dup2`），仅干净 JSON 走 stdout，彻底规避。
- **ONNX Runtime 可用（绕过 oneDNN）**：`--engine onnxruntime` + `CPUExecutionProvider`，
  不报 904、识别正确。**关键点**：ONNX 在 paddle **3.2.1 与 3.3.1（最新）都可用**；而 3.3.1
  的 MKLDNN 会崩溃，故 **想在 3.3.1 用最新 paddle 只能切 ONNX 才能正常出结果**
  （ONNX 绕过 oneDNN、不崩）。**GUI「推理引擎」下拉可切 Paddle(MKLDNN) / ONNX Runtime。**
  同图（1184×554）本机自测，均同 venv 3.2.1、同线程(6)/边长(1920)，仅后端不同：

  | 后端 | 冷启动(含模型加载) V6/V4 | 常驻warm(仅推理) V6/V4 |
  |---|---|---|
  | MKLDNN@3.2.1（默认） | 13.6s / 10.1s | 7.96s / 7.34s |
  | ONNX@3.2.1 | 9.6s / 9.0s | **7.21s / 5.69s** |
  | ONNX@3.3.1 | 10.8s / 11.0s | （venv 已清，未补） |

  **速度结论（同口径对比）**：无论冷启动还是常驻 warm，**ONNX ≤ MKLDNN**——
    warm 下 ONNX 反而更快（V4 5.69s vs 7.34s，快 ~22%）。**"ONNX 比 MKLDNN 慢/只是更慢的兜底"是错的**
    （旧结论源于误把「warm ~8s」与「ONNX 冷启动 ~10.8s」相减，口径不一致）。
  真正决定选型的是**兼容性**而非速度：3.2.1 的 MKLDNN 稳定、作默认；
    3.3.1 的 MKLDNN 会崩，只能切 ONNX。两者纯 CPU、速度同量级，随需切换。
  OpenVINO EP 本机不可用。
- **语言 → 模型代自动回退**：PP-OCRv6 覆盖 简/繁/英/日 + 拉丁语系；**韩文、俄文 v6 不支持，
  自动回退到 PP-OCRv5**（同为本代最新多语模型，识别正常）。
- **俄文键是 `ru`**：paddleocr 3.x 的俄文用户态 lang 码是 `ru`（内部映射 `cyrillic`），
  旧文档写的 `cyrillic` 会报错，本插件已做映射。
- **CPU 推理速度**：纯 CPU，大图较慢属正常；可在设置里调大「限制图像边长」或「线程数」。

## 部署（选择 GPU ONNX Runtime + CUDA 版本）
**首次使用必须先部署依赖**（创建 `.venv_gpu` 并装入 paddle + paddleocr + onnxruntime-gpu）：

- **推荐：双击根目录 `setup.bat`** → 两段式部署：第 1 步选模型范围，第 2 步选推理后端（GPU / CPU），选 GPU 时第 3 步追问 CUDA 版本：
  - `[1]` onnxruntime-gpu **1.27.0 + CUDA 13**（新显卡 / 驱动 R580+）
  - `[2]` onnxruntime-gpu **1.26.0 + CUDA 12.9**（RTX 30/40 系，驱动 ≥535）← 默认
  - `[3]` 纯 CPU onnxruntime 1.27.0（不用 GPU）
  - 脚本自动建 `.venv_gpu`、安装、并验证 providers（含 CUDAExecutionProvider 即 GPU 就绪）。
  - 之后在 Umi-OCR 全局设置 → 推理引擎 选对应项即可。
- 升级 / 切换 CUDA 版本：重跑 `setup.bat` 选另一个编号，`pip --upgrade` 会自动替换。
- ⚠️ 若曾用 `uv` 在 git-bash 里以 `/c/...` 路径装依赖，uv 会把路径错拼成 `C:\c\...`
  产生重复目录（见项目记忆），请用 Windows 原生路径或 `部署.bat` 避免。

## 重建 venv（如需，等效于 部署.bat 的 [2]）
```bat
cd Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py
uv venv --python 3.11 .venv_gpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
（用 Windows 原生路径调用 uv；git-bash 的 `/c/...` 会被错拼成 `C:\c\...`。）

## 日志与后端标注（排查用）
引擎每次启动与每次识别都会在 **stderr** 打印，并写入插件目录 `engine_active.log`：
- 启动：`init_ok ... backend=gpu(onnx-cuda)/cpu(onnx)/cpu(paddle) device=gpu/cpu onnxruntime=x.y.z (/CUDA 12.x) [⚠回退CPU]`
- 每次识别：`[engine] #N {耗时}s ... backend=... ver=PP-OCRv6 conf={平均置信度:.2f} out_code=... n_text=...`
- **选中 GPU 引擎但本机 CUDA 不可用时会明确标 `[⚠回退CPU]` 并打 WARN**，功能仍正常（仅无 GPU 提速）；
  若需真 GPU，重跑 `setup.bat` 选 [1]/[2] 装 onnxruntime-gpu。

## 验证记录
端到端跑通：六种语言经真实入口 `run.cmd`（`--config_path models/config_*.txt`）均正确输出
Umi-OCR 格式 JSON；`image_path` 与 `image_base64` 两种输入均正常。
详见 `../../../../docs/01_分析与设计.md` §11。
