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
| `requirements.txt` | `paddlepaddle==3.3.1` + `paddleocr==3.7.0` |
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
- **MKL-DNN 已强制关闭**：paddle 3.3.1 的 oneDNN 在当前 Windows/CPU 构建下推理期崩溃
  （`ConvertPirAttribute2RuntimeAttribute ... onednn`），故引擎忽略 Umi-OCR 下发的 `enable_mkldnn`
  （即使 GUI 勾选也无效），走原生 CPU。功能正常，仅少了 MKLDNN 加速。
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
