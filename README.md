# umi-paddle-neoengine

**Umi-OCR 的新一代 PaddleOCR 本地引擎插件**（Route B：Python 插件调用官方 `paddleocr.PaddleOCR()`，主程序零改动）。

本仓库**只放插件源代码**。安装包 / 纯净部署 zip / 懒人包请见本地发布目录 `Local-Ocr_发布包/`，**不进本 git 仓库**。

远端：<https://github.com/chapterv/umi-paddle-neoengine>

---

## 这是什么

| 项目 | 说明 |
|------|------|
| 目标软件 | [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) v2.1.5（框架冻结） |
| 插件目录名 | `win_x64_PaddleOCR_Py` |
| 引擎能力 | PP-OCR **v6 / v5 / v4**，多语言（简/繁/英/日/韩/俄） |
| **默认推理** | **ONNX Runtime · CPU**（`--engine onnxruntime`）——开箱优先能跑 |
| 可选 | ONNX CUDA GPU / Paddle+MKLDNN CPU（GUI「推理引擎」切换） |
| 关键版本钉死 | **`paddlepaddle==3.2.1`** + `paddleocr==3.7.0` |

单一插件、单一 `engine.py`，三种后端靠 GUI / `--engine` 切换：

| 后端 | 说明 |
|------|------|
| **ONNX Runtime · CPU**（默认） | `onnxruntime` / `CPUExecutionProvider`。部署路径含中文时也更稳。 |
| **ONNX Runtime · CUDA GPU** | `onnxruntime-gpu`；不可用时自动回退 CPU。推荐 `onnxruntime-gpu[cuda,cudnn]==1.26.0`（CUDA12 pip 自带 DLL，**无需系统 CUDA Toolkit**）。 |
| **Paddle (MKLDNN) · CPU** | 原生 + oneDNN；必须 `paddlepaddle==3.2.1`（3.3.x 的 MKLDNN 会崩）。 |

### 为何 paddle 必须 3.2.1？

`paddlepaddle` **3.3.x** 在 Windows/CPU 上开 MKLDNN 时，`predictor.run()` 可能崩溃（`ConvertPirAttribute2RuntimeAttribute` / PIR）。本仓库锁定 **3.2.1**。

### 其它关键修复

- 推理期 oneDNN 诊断污染 stdout → 临时重定向，JSON 协议干净。
- 语言映射：俄文用户态 `ru`；韩/俄无 V6 rec 时回退 V5。
- Windows 非 ASCII 路径（如「发布包」）：Paddle 原生读模型失败时，缓存目录改 8.3 短路径。
- `run.cmd` 优先选择同时有 `paddleocr` + `onnxruntime` 的 venv。

---

## 目录结构（仅源码）

```text
umi-paddle-neoengine/
├── README.md
├── .gitignore
└── win_x64_PaddleOCR_Py/
    ├── engine.py
    ├── PPOCR_umi.py / PPOCR_api.py / PPOCR_config.py
    ├── run.cmd
    ├── requirements.txt   # paddle 3.2.1 + paddleocr 3.7.0 + onnxruntime-gpu[cuda,cudnn]==1.26.0
    ├── i18n.csv
    ├── models/            # 语言 config_*.txt
    └── README.md
```

**不包含**：`.venv/`、`paddlex/`、日志、zip 发布包。

---

## 安装到 Umi-OCR

1. 将 `win_x64_PaddleOCR_Py/` 复制到 `UmiOCR-data/plugins/`。
2. 用发布包根目录 `setup.bat` 装依赖（推荐），或手动：

   ```bat
   cd Umi-OCR\UmiOCR-data\plugins\win_x64_PaddleOCR_Py
   uv venv --python 3.11 .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   纯 CPU 也可：`pip install paddlepaddle==3.2.1 paddleocr==3.7.0 onnxruntime==1.26.0`
3. Umi-OCR → 全局设置 → 文字识别 → 选本插件；默认引擎为 **ONNX CPU**。

---

## 许可

插件本体遵循 Umi-OCR 相关许可；修改与打包按各自仓库约定。
