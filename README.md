# umi-paddle-neoengine

**Umi-OCR 的新一代 PaddleOCR 本地引擎插件**（Route B：Python 插件调用官方 `paddleocr.PaddleOCR()`，无需重编 C++）。

本仓库**只放插件源代码**。安装包 / 纯净部署 zip / 懒人包请见本地发布目录 `Local-Ocr_发布包/`，**不进本 git 仓库**。

远端：<https://github.com/chapterv/umi-paddle-neoengine>

---

## 这是什么

| 项目 | 说明 |
|------|------|
| 目标软件 | [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) v2.1.5（框架冻结，主程序零改动） |
| 插件目录名 | `win_x64_PaddleOCR_Py` |
| 引擎能力 | PP-OCR **v6 / v5 / v4**，多语言（简/繁/英/日/韩/俄） |
| 默认推理 | **Paddle + MKLDNN（oneDNN）CPU 加速** |
| 关键版本钉死 | **`paddlepaddle==3.2.1`** + `paddleocr==3.7.0` |

### 为何必须 3.2.1？

`paddlepaddle` **3.3.x** 在 Windows/CPU 上开启 MKLDNN 时，会在 `predictor.run()` 崩溃：

```text
ConvertPirAttribute2RuntimeAttribute ... pir::ArrayAttribute
```

官方相关讨论：PaddlePaddle#77340 / PaddleOCR#17869。

本仓库通过 **降级并锁定 3.2.1** 使 **MKLDNN 可稳定开启**，实测相对关 MKLDNN 约 **2～3.5×** 提速（视图尺寸与版本而定）。

### 其它已合入的关键修复（首发里程碑）

- **Path-1 / 904**：推理期 oneDNN 往 stdout 打诊断信息污染 JSON → `engine.py` 临时 `stdout→stderr`，协议行干净。
- **语言映射**：俄文用户态码为 `ru`（不是直接传 `cyrillic`）。
- **韩/俄与 V6**：V6 无韩/俄 rec 时自动回退到 V5 系模型（后续迭代还会继续加强）。

---

## 目录结构（仅源码）

```text
umi-paddle-neoengine/
├── README.md                 ← 本文件
├── .gitignore
└── win_x64_PaddleOCR_Py/     ← 丢进 Umi-OCR 的 plugins 目录即可
    ├── __init__.py
    ├── engine.py             # worker：PaddleOCR 推理 + 协议
    ├── PPOCR_umi.py          # Umi-OCR class Api 契约
    ├── PPOCR_api.py          # stdin/stdout JSON 管道客户端
    ├── PPOCR_config.py       # GUI 全局选项（含 MKLDNN 开关）
    ├── run.cmd               # 入口：.venv\python engine.py
    ├── requirements.txt      # paddlepaddle==3.2.1 + paddleocr==3.7.0
    ├── i18n.csv
    ├── models/               # 仅语言下拉占位 config_*.txt
    └── README.md             # 插件内补充说明
```

**不包含**：`.venv/`、`paddlex/` 模型缓存、日志、zip 发布包。

---

## 安装到 Umi-OCR

1. 将本仓库中的 **`win_x64_PaddleOCR_Py/` 整个文件夹**复制到：

   ```text
   <Umi-OCR>/UmiOCR-data/plugins/win_x64_PaddleOCR_Py/
   ```

2. 在该目录创建虚拟环境并安装依赖（推荐 **Python 3.10 / 3.11**）：

   ```bat
   cd UmiOCR-data\plugins\win_x64_PaddleOCR_Py
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -U pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. 启动 Umi-OCR → **全局设置 → 文字识别** → 选择本插件（名称类似 **PaddleOCR（本地）**）。

4. 首次识别会按语言自动下载模型（默认缓存到 paddlex 相关目录，体积视语言而定）。

### 可选：一键部署包

若你持有配套发布 zip（不在本仓库）：

| 包名（示例） | 用途 |
|--------------|------|
| `Local-Ocr_纯净部署版.zip` | 仅程序+脚本，自建 venv |
| `Local-Ocr_ONNX_V6_CPU版.zip` | 懒人包：ONNX V6 模型 + 最小 CPU 环境 |

发布包放在本机 **`Local-Ocr_发布包`** 目录，**不要**提交进 git。

---

## MKLDNN / 加速说明（本阶段重点）

| 项 | 建议 |
|----|------|
| paddle 版本 | **固定 3.2.1**，勿升 3.3.x |
| GUI「启用 MKL-DNN」 | 默认开；3.2.1 下可加速 |
| 若仍崩溃 | 关 MKLDNN 或后续版本使用 ONNX 后端（见后续提交） |
| 线程数 / 边长 | 可在全局设置调节以平衡速度与精度 |

本机参考口径（长截图约 1184×554，常驻引擎热推理量级）：V6+MKLDNN 约数秒～十余秒量级，视 CPU 而定；详见插件内 `README.md` 与历史实测记录。

---

## 开发与版本节奏

- **本远端 `main` 首发**：推送到 **MKLDNN CPU 引擎可用且加速修复完成** 的里程碑（含 Path-1 + 3.2.1 锁定 + 实测口径订正）。
- 本地分支 `archive/full-history` / `master` 保留其后全部历史（ONNX 切换、GPU、坐标对齐、回退链等），**后续再按批次推送**。
- 完整产品工程（Umi 主程序、docs、打包脚本）仍在本地 monorepo `Local-Ocr`。

---

## 协议与归属

- 插件对接 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) 插件契约。
- 算法依赖 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / PaddlePaddle。
- 本仓库改动与打包约定以本 README 与提交历史为准。

---

## 快速自检

```bat
cd win_x64_PaddleOCR_Py
.venv\Scripts\python.exe -c "import paddle; import paddleocr; print(paddle.__version__, paddleocr.__version__)"
```

期望：`3.2.1` 与 `3.7.0`（或兼容的 3.7.x）。然后启动 Umi-OCR 选本引擎，识别一张中文图应返回 `code:100` 的 JSON 结果。
