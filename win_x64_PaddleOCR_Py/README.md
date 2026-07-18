# win_x64_PaddleOCR_Py —— Umi-OCR 的 PaddleOCR（Python）引擎插件

基于 **PaddleOCR 3.x（pipeline 架构，PP-OCRv6/v5）** 的本地 OCR 引擎，
以 Umi-OCR 插件形式提供，与官方 `hiroi-sora/PaddleOCR-json` 管道协议完全兼容，
**Umi-OCR 主程序零改动**。

> 路线：自写 Python 插件（Route B），引擎子进程直接调用官方 `paddleocr.PaddleOCR()`
> 推理，复用 Umi-OCR 自带的 `PPOCR_umi` / `PPOCR_api` JSON 管道。
> 旧引擎 `win7_x64_PaddleOCR-json` 保留作为回退，本插件是「新增可选引擎」。

## 文件说明
| 文件 | 作用 |
|---|---|
| `run.cmd` | 引擎入口。Umi-OCR 调它 → 用本目录 `.venv` 的 python 跑 `engine.py` |
| `engine.py` | **worker**：解析参数 → 建 `PaddleOCR` → `OCR init completed.` 握手 → 逐行读 JSON 识图 |
| `PPOCR_api.py` | 管道客户端（从 Umi-OCR 自带 Python 运行，不依赖 paddle） |
| `PPOCR_umi.py` | Umi-OCR 插件接口 `class Api`（标准契约） |
| `PPOCR_config.py` / `i18n.csv` | 引擎设置面板（语言下拉框、方向分类、边长限制等） |
| `models/configs.txt` + `config_*.txt` | **仅驱动语言下拉框**，引擎不读其内容；真实模型由 paddle 自动下载 |
| `requirements.txt` | `paddlepaddle==3.2.1` + `paddleocr==3.7.0` |
| `.venv/` | 隔离 Python 环境（含 paddle，已装；**不进 git**） |

## 使用方法（Umi-OCR GUI）
1. 用 **Umi-OCR v2.1.5**（或兼容版本）打开本项目 `Umi-OCR/` 目录的程序。
2. 全局设置 → 「引擎」下拉框，选择本插件（名为 **PaddleOCR（本地）** / 对应 `win_x64_PaddleOCR_Py`）。
3. 「文字识别 → 语言/模型库」选择所需语言：
   - 简体中文 / English / 繁體中文 / 日本語 / 한국어 / Русский（俄文）
4. 首次识别某语言时，paddle 会**自动下载并缓存**对应模型到
   `C:\Users\<你>\.paddlex\official_models\`（约几十~上百 MB，只需一次）。
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
  （ONNX 绕过 oneDNN、不崩）。
  同图（1184×554）本机自测，均同线程/边长、仅后端不同：
    · MKLDNN@3.2.1（默认）：冷启动(含模型加载) V6≈13.6s / V4≈10.1s；
      常驻引擎单图推理(不含加载) V6≈8.0s / V4≈7.3s。
    · ONNX@3.3.1：冷启动 V6≈10.8s / V4≈11.0s（@3.2.1 为 V6≈9.6s / V4≈9.0s）。
  **速度结论（同口径对比）**：同为「冷启动」口径时 ONNX 与 MKLDNN 在同一量级，
    V6 上 ONNX 略快、V4 上基本持平——**"ONNX 比 MKLDNN 慢"是错的**
    （此前误把「UI 热推理 ~8s」与「ONNX 冷启动 ~10.8s」相减，口径不一致）。
  真正决定选型的是**兼容性**而非速度：3.2.1 的 MKLDNN 是稳定默认；
    3.3.1 的 MKLDNN 会崩，只能切 ONNX（速度相当，并非更慢的兜底）。
  （ONNX 常驻单图 warm 数字因临时 3.3.1 测试 venv 已清理未补测；需要可重建再测。）
  OpenVINO EP 本机不可用。
- **语言 → 模型代自动回退**：PP-OCRv6 覆盖 简/繁/英/日 + 拉丁语系；**韩文、俄文 v6 不支持，
  自动回退到 PP-OCRv5**（同为本代最新多语模型，识别正常）。
- **俄文键是 `ru`**：paddleocr 3.x 的俄文用户态 lang 码是 `ru`（内部映射 `cyrillic`），
  旧文档写的 `cyrillic` 会报错，本插件已做映射。
- **CPU 推理速度**：纯 CPU，大图较慢属正常；可在设置里调大「限制图像边长」或「线程数」。

## 重建 venv（如需）
```bat
cd Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
（用 Windows 原生路径调用 uv；git-bash 的 `/c/...` 会被错拼成 `C:\c\...`。）

## 验证记录
端到端跑通：六种语言经真实入口 `run.cmd`（`--config_path models/config_*.txt`）均正确输出
Umi-OCR 格式 JSON；`image_path` 与 `image_base64` 两种输入均正常。
详见 `../../../../docs/01_分析与设计.md` §11。
