# umi-paddle-neoengine

**Umi-OCR 本地 PP-OCRv6 引擎插件**（Route B：Python 插件调用官方 PaddleOCR 3.x）

[![Version](https://img.shields.io/badge/version-1.3-orange)](./VERSION)
[![Umi-OCR](https://img.shields.io/badge/Umi--OCR-v2.1.5-blue)](https://github.com/hiroi-sora/Umi-OCR)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.7-green)](https://github.com/PaddlePaddle/PaddleOCR)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/github-chapterv%2Fumi--paddle--neoengine-black)](https://github.com/chapterv/umi-paddle-neoengine)

面向 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) 的**本地离线**新引擎插件：在**不改主程序**的前提下，把识别能力从内置老旧 PP-OCRv3，升级到官方 **PaddleOCR 3.x（PP-OCRv6 / v5 / v4）**，并支持 **ONNX CPU / ONNX CUDA GPU / Paddle+MKLDNN** 三种推理后端。

- **当前源码版本**：**1.3**（见仓库根目录 [`VERSION`](./VERSION)；开发以 **`master`** 为准）
- **本仓库（源码）**：<https://github.com/chapterv/umi-paddle-neoengine>  
- **完整发布包**（含 Umi 主程序 + `setup.bat`）：同级目录 **`umi-paddle-neoengine-release/`**（zip **不进**本 git 仓库）  
  - `umi-paddle-neoengine-deploy-v1.3.zip` — 纯净部署（需 `setup.bat`）
  - `umi-paddle-neoengine-ONNX-V6-CPU-v1.3.zip` — ONNX V6 CPU 懒人包
- **宿主补丁（主程序 py_src 修复）**：[`patches/umi-host/`](./patches/umi-host/)
  - 完整 zip 已内嵌；若只装插件、主程序仍是官方原版，请运行
    [`patches/umi-host/apply_host_patches.bat`](./patches/umi-host/apply_host_patches.bat)
    （可拖入 `Umi-OCR` 目录；会先备份再覆盖 25 个宿主/插件文件）

---

## 项目简介

### 关于 Umi-OCR（原项目）

**Umi-OCR** 是一款免费、开源、可批量的离线 OCR 软件，界面友好、插件化、适合截图与批量识图：

- 完全免费、无广告  
- 现代化图形界面，批量处理  
- 多语言、插件扩展  
- Windows 等平台可用  

官网仓库：<https://github.com/hiroi-sora/Umi-OCR>

#### 痛点

Umi-OCR 本体长期自带的本地引擎仍是 **PaddleOCR-json + 较老的 PP-OCRv3** 路线：

| 问题 | 说明 |
|------|------|
| **引擎代际落后** | 上游 PaddleOCR 已演进到 3.x / PP-OCRv6，官方 GUI 主线更新节奏与模型代际不同步 |
| **难字 / 手写 / 残缺图** | v3 时代模型在复杂手写、低质样张上漏检、错字明显 |
| **多语言质量** | 日文、韩文等段落在老模型上易丢标点、结构错乱 |
| **性能路径单一** | 缺少开箱即用的 **ONNX-CUDA GPU** 加速与现代后端切换 |

社区有 [AI 云端 OCR 插件](https://github.com/EatWorld/UmiOCR-AI-OCR-Plugin) 用多模态大模型补精度（效果好，但依赖网络、API 与费用）。本项目走另一条路：**继续本地、离线、免费**，把引擎升级到 v6 并补齐部署与性能。

### 本项目是什么

本仓库提供插件目录 **`win_x64_PaddleOCR_Py/`**：

- 以 Umi 标准插件契约接入（`class Api` + stdin/stdout JSON）  
- 子进程调用官方 **`paddleocr==3.7.0` + `paddlepaddle==3.2.1`**  
- **默认推荐：ONNX Runtime CPU**（开箱稳、路径含中文也更不易踩坑）  
- 可选：**ONNX Runtime CUDA GPU**（同图热身约 **1.5～2s** 级，视预处理开关而定）  
- 可选：**Paddle 原生 + MKLDNN**（CPU 加速，须钉死 3.2.1）  
- 模型缓存到插件内 **`paddlex/`**，懒人包可预置、纯净包可首次自动下载  
- P0 几何表格 CSV 默认可用；P1 复杂表结构模型按需安装、默认关闭
- 官方原版 Umi 如需表格 UI/导出，运行 [`patches/umi-host/apply_host_patches.bat`](./patches/umi-host/apply_host_patches.bat)

一句话：**Umi 壳子不动，本地识别引擎换代到 PP-OCRv6，并可选 GPU。**

---

## 对比识别效果

> 下列素材均在 `images/` 目录；内容按识别效果优先、性能对比最后的顺序展示。

### 手写样张 1（对照云端 AI 案例）

参考社区 [UmiOCR-AI-OCR-Plugin](https://github.com/EatWorld/UmiOCR-AI-OCR-Plugin) README 中的对比叙事：  
**复杂手写** 在旧版本地 Paddle / WeChatOCR 上往往很差，而 **Gemini 等云端视觉模型** 可以接近完美。

本项目用 **本地 PP-OCRv6** 跑同类手写样张，说明：**不必上云，v6 本地已可达到接近可用的手写效果。**

**原图：**

![样章-手写1](images/样章-手写1.png)

**本插件 PP-OCRv6 结果：**

![手写1-v6](images/手写1-v6.png)

---

### 手写样张 2（对照 AI：仅个别错字）

与云端 AI 精细结果相比，本地 v6 在本样张上已非常接近，**仅个别错字**，适合「离线 + 够用精度」场景。

**原图：**

![样章-手写2](images/样章-手写2.png)

**本插件 PP-OCRv6 结果：**

![手写2-v6](images/手写2-v6.png)

> 若业务要求「零容错、任意潦草手写」，仍可叠加 [AI OCR 插件](https://github.com/EatWorld/UmiOCR-AI-OCR-Plugin) 作第二通道；日常离线办公，本地 v6 通常足够。

---

### 日语段落

**原图：**

![样章-日语段落](images/样章-日语段落.png)

| 引擎 | 结果 |
|------|------|
| 旧 **v3** | ![日语 v3](images/样章-日语段落v3.png) |
| 本插件 **v6** | ![日语 v6](images/样章-日语段落v6.png) |

**v6 无需手工调整即可正确识别** 日文段落主体（假名/汉字混排稳定）。语言请在 Umi 中选 **日本語**（或对应 config）。

---

### 韩语段落

**原图：**

![样章-韩语段落](images/样章-韩语段落.png)

| 引擎 | 结果 |
|------|------|
| 旧 **v3** | ![韩语 v3](images/样章-韩语段落v3.png) |
| 本插件 **v6**（实际走 v5 韩文 rec，见下） | ![韩语 v6](images/样章-韩语段落v6.png) |

**说明：** PP-OCRv6 官方 **无韩文 rec** 时，本插件会按回退链切到 **PP-OCRv5 韩文模型**；界面仍可选「优先 v6」。  
上图 **v6 链路结果中，标点符号识别正确**，整体可读性优于旧 v3。请将语言选为 **한국어**（不要用「简体中文」硬识韩文，否则易出 `?`）。

---

### 表格识别效果：原图、V3 / V6 与 Excel

下面用同一张出入库表做对比。先看原图，再看旧 PP-OCRv3 与本插件 PP-OCRv6
的识别结果；每一列下方都是把对应结果粘贴进 Excel 后的样子。

**原始样张：**

![表格原始样张](images/样章-表格.png)

| 旧 PP-OCRv3 | 本插件 PP-OCRv6 |
|---|---|
| ![表格 v3 识别结果](images/样章-表格v3.png) | ![表格 v6 识别结果](images/样章-表格v6.png) |
| **粘贴到 Excel 后**<br>![表格 v3 Excel](images/样章-表格v3-excel.png) | **粘贴到 Excel 后**<br>![表格 v6 Excel](images/样章-表格v6-excel.png) |

> 这组图用于直观看识别文本和表格落格的差别。规整表格优先使用 P0 几何方式；
> 合并单元格、无线框等难表再按需开启 P1 结构模型。

---

### 残缺 / 低质样张

**原图：**

![样章-残缺识别](images/样章-残缺识别.png)

**本插件 v6 识别结果：**

![残缺识别1](images/残缺识别1.png)

低质、残缺印刷在 v6 下仍能稳住主体文字，明显优于老 v3 引擎常见「大片空白 / 乱码」。

---

### 性能识别样章（同图 · 后端对比）

同一张性能测试图，不同引擎/后端的界面截图（文件名中含大致耗时，作对照用；实际以本机为准）：

| # | 说明 | 截图 |
|---|------|------|
| **0** | 旧链路 **PP-OCRv3 + MKL 开启** ≈ 6.44s | ![0-v3-MKL](images/0-v3-MKL开启-6.44.png) |
| **1** | **PP-OCRv6 + MKL 未开** ≈ 63s（极慢，勿用） | ![1-v6-no-mkl](images/1-v6-MKL未开-63.09.png) |
| **2** | **PP-OCRv6 + MKL 开启** ≈ 15s | ![2-v6-mkl](images/2-v6-MKL开启-15.29.png) |
| **3** | **PP-OCRv6 + ONNX CPU** ≈ 11.6s | ![3-onnx-cpu](images/3-v6-ONNX-CPU-11.56.png) |
| **4** | **PP-OCRv6 + ONNX GPU** ≈ **1.91s** | ![4-onnx-gpu](images/4-v6-ONNX-GPU-1.91.png) |

**读图结论（人话）：**

- v6 比 v3 **质量更好**，但若 MKLDNN 关错或环境不对，会慢到不可用（图 1）。
- 纯 CPU 上 v6 常见 **十余秒** 量级（图 2/3）；要「秒级」需 **ONNX + CUDA**（图 4）。
- GUI 里若打开 **文档去扭曲 / 方向纠正** 等预处理，GPU 也会从 ~1.9s 涨到 ~3s，属正常（多跑模型，不是 GPU 失效）。

---

## 本项目详细功能特点

| 功能 | 描述 |
|------|------|
| **引擎代际** | PP-OCRv6 / v5 / v4，可配置回退链（如 V6→V5→V4） |
| **三后端合一** | 同一插件内切换：`onnxruntime`（默认）/ `onnxruntime-gpu` / `paddle`（MKLDNN） |
| **GPU 加速** | ONNX + CUDA EP；缺 CUDA 时自动回退 CPU 并日志提示 |
| **模型自包含** | `PADDLE_PDX_CACHE_HOME` → 插件内 `paddlex/`，懒人包可预置 |
| **多语言** | 简/繁/英/日/韩/俄等（韩/俄等无 v6 rec 时自动降级 v5） |
| **文档预处理** | 方向纠正 / 去扭曲(UVDoc) / 行方向（可关以换速度） |
| **协议兼容** | 兼容 Umi `PPOCR_api` JSON 管道；修复 oneDNN 污染 stdout 导致的 904 |
| **中文路径** | Paddle 原生在「发布包」等中文路径下易失败；已做 8.3 短路径规避 |
| **P0 几何表格** | OCR 坐标自动重建行列；输出 UTF-8 BOM 结构 CSV，无额外模型 |
| **P1 结构模型** | `TableRecognitionPipelineV2`；复杂有线/无线表和合并表头；失败回退 P0 |
| **部署脚本** | `setup.bat` 安装基础 OCR 并可选 P1；`install_table_models.bat` 供老用户补装 |
| **双发布形态** | 纯净包（小、需 setup）/ ONNX V6 CPU 懒人包（大、含 venv+模型） |

---

## 安装要求

1. **Umi-OCR**：推荐 [Umi-OCR v2.1.5](https://github.com/hiroi-sora/Umi-OCR)（本项目按该版本契约测试）  
2. **系统**：Windows x64  
3. **Python**（纯净部署 / 自建环境时）：**3.11+**（GPU 装 `onnxruntime-gpu` 1.26+ 必需；3.10 最高只能到较旧 GPU 轮子）  
   - 推荐安装 [uv](https://docs.astral.sh/uv/)，`setup.bat` 可自动拉取 CPython 3.11  
4. **GPU 可选**：NVIDIA 显卡 + 较新驱动。  
   - **推荐**：`onnxruntime-gpu[cuda,cudnn]==1.26.0`（CUDA 12.x DLL **进 venv**，一般**不必**再装系统 CUDA Toolkit）  
   - `1.27 + CUDA13` 的 pip extras 目前在 PyPI 上 cu13 零件不全，不推荐当默认  
5. **磁盘 / 网络**：纯净包首次需下载 paddle/模型；懒人包体积更大、可离线用  

---

## 安装步骤

### 方式 A：完整发布包（推荐最终用户）

1. 从 `umi-paddle-neoengine-release/` 取 zip：  
   - **`umi-paddle-neoengine-deploy-v1.3.zip`**：小包，需联网 `setup.bat`
   - **`umi-paddle-neoengine-ONNX-V6-CPU-v1.3.zip`**：含 `.venv` + V6 ONNX 模型，默认 ONNX CPU
2. 解压到**尽量纯英文路径**（如 `C:\Local-Ocr\`；中文路径下 paddle 原生更易出问题，ONNX 相对稳）  
3. 纯净包：双击根目录 **`setup.bat`**  
   - 模型范围：默认「最小可用 = 中文 V6 ONNX」即可  
   - 推理后端：默认 **[2] 纯 CPU**；有 N 卡选 **[1] GPU**，第 3 步默认 **1.26 + CUDA12.9**  
   - 第 4 步询问 P1 表格结构模型；直接回车默认跳过，需要时输入 `Y`
4. 运行 `Umi-OCR\Umi-OCR.exe`  
5. **全局设置 → 文字识别**：引擎选本插件（PaddleOCR 新引擎）  
6. 推理引擎建议：  
   - 无 GPU / 要稳 → **ONNX Runtime CPU**  
   - 有 N 卡要快 → **ONNX Runtime CUDA GPU**（并视需要关闭「文档去扭曲」以接近 ~2s）  

已有基础环境，只补装 P1：

```bat
install_table_models.bat
```

脚本会按 `run.cmd` 相同顺序选择 `.venv_gpu → .venv`。使用 `--check` 可只检查，
`--deps-only` 可只安装依赖、把模型下载延后到首次启用。

### 方式 B：仅插件源码（开发者）

1. 将本仓库 `win_x64_PaddleOCR_Py/` 复制到：  
   ```text
   <Umi-OCR>/UmiOCR-data/plugins/win_x64_PaddleOCR_Py/
   ```
2. 在该目录建 venv 并安装依赖，例如：  
   ```bat
   cd UmiOCR-data\plugins\win_x64_PaddleOCR_Py
   uv venv --python 3.11 .venv
   .venv\Scripts\python.exe -m pip install paddlepaddle==3.2.1 paddleocr==3.7.0 onnxruntime==1.26.0
   rem GPU 可选：
   rem .venv\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
   ```
3. 重启 Umi-OCR，选择本插件；首次识别会向插件内 `paddlex/` 拉模型。  

---

## 表格识别与结构 CSV

### P0 几何方式（默认可用）

P0 不安装新模型，适合边框清楚、行列规整的表：

1. 打开 **批量识图**或**批量文档**。
2. 右侧进入 **设置**。
3. 展开 **保存文件类型**，勾选
   **`table.csv 结构表格(Excel)`**。
4. 开始任务。系统会在普通排版前使用原始 OCR 坐标自动建立二维表。
5. 如果还要在结果区预览 TSV 表格，在
   **OCR文本后处理 → 排版解析方案**选择
   **`表格-几何网格`**。

**设置 C：在「OCR文本后处理 → 排版解析方案」中选择 `表格-几何网格`。**

![设置 C：选择表格-几何网格排版](images/设置C.png)

输出遵循页面的 **保存到** 设置，文件名以 `_table.csv` 结尾，编码为 UTF-8
BOM，可直接用 Excel 打开。

**设置 A：在批量 OCR 的「保存文件类型」中勾选 `table.csv`。**

![设置 A：勾选 table.csv 结构表格输出](images/样章-表格设置A.png)

### P1 模型方式（可选）

P1 适合复杂有线/无线表和合并表头。先通过 `setup.bat` 第 4 步输入 `Y`，或运行
`install_table_models.bat`；额外模型约 955 MB。安装后：

1. 完全退出并重启 Umi-OCR。
2. 进入 **全局设置 → 文字识别**。
3. 确认 **当前接口**是本项目 PaddleOCR 本地新引擎。
4. 开启 **`表格结构模型（P1·可选）`**。
5. 点击绿色按钮 **`应用修改`**；看到“文字识别接口应用成功”后生效。
6. 回到批量识图/批量文档，勾选
   **`table.csv 结构表格(Excel)`**输出。

**设置 B：在「全局设置 → 文字识别」中开启 P1 表格结构模型，并点击「应用修改」。**

![设置 B：开启 P1 表格结构模型](images/样章-表格设置B.png)

| 项目 | P0 几何方式 | P1 结构模型 |
|------|-------------|-------------|
| 安装 | 默认已有 | 可选依赖 + 约 955 MB 模型 |
| 原理 | 文本框坐标聚类 | SLANeXt/单元格检测 + OCR 填格 |
| 速度/内存 | 轻量 | 冷启动和内存明显更高 |
| 规整表 | 推荐 | 可用 |
| 无线/复杂表 | 有限 | 通常更好 |
| 合并单元格 | CSV 展开/近似 | 保留 HTML rowspan/colspan 后展开到 CSV |
| 失败处理 | 无模型失败点 | 自动回退 P0，不破坏普通 OCR |

---

## 竖排 PDF 与复制顺序

- **双层 `.layered.pdf`**：无需开关。系统自动判断横排/竖排文字框，每个竖排
  OCR 栏写入一个标准嵌入 CJK 连续 run。Chrome/Edge 可局部高亮，复制单栏时
  文字连续；支持 0°/90°/180°/270° 页面和轻微倾斜四点框。
- **跨栏限制**：Chrome/Edge 跨多个独立竖栏拖选时仍按几何左→右且可能插入
  换行。Tagged PDF、`ActualText` 与 W2 单流都无法同时保留局部高亮并重排
  剪贴板。需要严格连续语序时，请选单层 `统一横排` 或同时导出 TXT。
- **单层 `.text.pdf`**：进入
  **批量文档 → 设置 → 保存文件类型**，勾选
  **`text.pdf 单层纯文本文档`** 后，在下方选择
  **单层 PDF 排版方向**：
  - `自动（保留 OCR 排版）`：默认，允许同页横竖混排。
  - `统一横排`：所有文字块按横向排版。
  - `统一竖排`：所有多字符文字块按纵向排版。
- 单层自动/统一竖排使用标准 CJK 纵向字体指标，汉字保持直立且文字完整。
  方向选项不会改变双层 PDF；双层始终自动对齐原始扫描图。
- 竖排 `。` 只在原图存在唯一高置信环形连通域，且尺寸、位置、视觉 cell 和
  低墨量条件同时通过时恢复；不会根据换行或语义猜标点。紧急回退可在启动前设置
  `UMI_OCR_VERTICAL_PUNCTUATION_RECOVERY=0`。

---

## 配置说明

| 配置项 | 建议 | 说明 |
|--------|------|------|
| **推理引擎** | 默认 `onnxruntime`；要快选 `onnxruntime-gpu` | 与「是否真在用 CUDA」以结果旁标签 / 日志为准 |
| **模型版本** | 中文优先 PP-OCRv6 | 韩/俄等会自动回退到有 rec 的版本 |
| **回退链** | 如 `PP-OCRv6,PP-OCRv5,PP-OCRv4` | 某版本初始化失败时依次尝试 |
| **文档方向 / 去扭曲 / 行方向** | 要速度可关；弯页扫描建议开 | 去扭曲（UVDoc）对耗时影响最大 |
| **限制边长** | 默认约 1920 | 过大更慢；过小损小字 |
| **CPU 线程** | 按机器核数 | 主要影响 paddle/CPU 路径 |
| **语言** | 与图片语种一致 | 韩文务必选韩语，勿用中文模型硬识 |
| **表格结构模型（P1·可选）** | 默认关闭 | 开启后需点击“应用修改”；用于复杂表，失败回退 P0 |
| **OCR 调试追踪 JSONL（可选）** | 默认留空 | 同页记录 raw_ocr / preview / document_export，用于核对标点证据 |

**如何确认 GPU 生效：**  
识别结果耗时旁或日志中出现 `gpu(cuda)` / `gpu(onnx-cuda)`；若出现回退 CPU 的 WARN，检查是否装了 `onnxruntime-gpu` 且驱动正常。

**速度对照（同性能测试图，本机示例）：**

| 条件 | 约热身耗时 |
|------|------------|
| ONNX GPU + 预处理关 | **~1.5–1.9s** |
| ONNX GPU + 预处理全开 | **~3.2s** |
| ONNX CPU / paddle CPU | **十余秒** 量级 |

---

## 更新日志

版本号写在仓库根目录 **`VERSION`**（当前 **`1.3`**）。

### v1.3（2026-07-24）

- **竖排 PDF 浏览器兼容**
  - 双层每个竖排 OCR 栏使用一个标准嵌入 CJK 连续 run；单栏可连续复制，
    并恢复旋转页、倾斜框及 `text + end` 实际标点。
  - Chrome/Edge 跨栏复制的几何左→右/换行作为已确认阅读器限制保留；
    严格语序使用单层统一横排或 TXT。
  - 单层 `text.pdf` 支持自动、统一横排、统一竖排；纵向字体保持汉字直立和
    完整提取。
- **竖排句号恢复**
  - 仅凭唯一高置信原图环形连通域恢复 `。`，记录来源、置信度、bbox 和插入位置；
    不从换行或语义推断，并提供环境变量回退开关。
- **P1 路由与性能**
  - 只有 `table.csv + P1` 明确请求才运行表格结构流水线；普通截图、预览和文档
    OCR 固定为 `task=ocr`。
  - GPU 短文本 ABBA 稳态实测：P1 能力开启但普通 OCR 的中位额外耗时 +1.717%。
  - 进一步表格识别精度优化列为远期 P2。
- **回归**
  - PDF 12/12、C3 5/5、P1 路由 4/4、引擎 8/8、追踪 5/5、
    宿主补丁 4/4、P0 表格 65/65。

### v1.2（2026-07-23）

- **新增 P0 几何表格导出**
  - 新增 `表格-几何网格` 排版和 `table.csv 结构表格(Excel)` 输出。
  - 结构 CSV 在普通排版前直接使用原始 OCR 坐标建表，不再从纯文本猜列。
  - CSV 使用 UTF-8 BOM，可直接用 Excel 打开。
- **新增 P1 可选结构模型**
  - 接入 `TableRecognitionPipelineV2`，复用 PP-OCRv6 文本块。
  - 支持 HTML 合并表头；模型失败自动回退 P0。
  - 默认关闭，约 955 MB 模型不进入基础安装。
- **安装与宿主补丁**
  - `setup.bat` 第 4 步可选安装 P1；直接回车默认跳过。
  - 新增 `install_table_models.bat` 老用户升级入口。
  - 宿主补丁从 5 个扩展到 15 个，覆盖表格解析、导出和 QML 开关。

### v1.1（批量文档不再“停在那儿不动”）

这一版主要解决批量文档里的两种卡住方式：

1. **空白页 / 异常页让整个任务停住。**
   - 问题：空白页没有任何文字框时，旧的排版预处理仍会计算 `median([])`；异常从文档
     worker 冒出去后，界面还以为任务在运行，后续文件也无法继续处理。
   - 修复：空文字框直接按空页返回；文档页的排版异常改为“记录这页错误、继续下一页”，
     不再让整批任务被一张坏页拖死。

2. **OCR 子进程没有返回，点“停止”后也无法再开始。**
   - 问题：文档任务在等待 OCR 结果时可能永久阻塞；旧的停止操作只改任务状态，正在等待的
     worker 没有机会检查状态，于是一直占着任务槽位。
   - 修复：等待改成可定时醒来的循环，单页超时会中止等待；强制恢复会清空队列、终止底层
     OCR 引擎并作废旧 worker，随后可以立即提交新任务。

- 同时统一发布包名称为 `deploy` 与 `ONNX-V6-CPU`，并提供宿主补丁安装器。

### v1.0（ONNX CUDA GPU 加速）

- 问题：即使电脑有 NVIDIA 显卡，旧流程也主要吃 CPU；PP-OCRv6 在大图上的等待时间较长。
- 修复：加入 ONNX Runtime CUDA 推理路径，安装时可选 GPU 环境；运行时会检查 CUDA 是否真的
  可用，缺少 DLL、驱动不匹配或初始化失败时自动回退 CPU，并在结果标签和日志中说明原因。
- 同一个插件因此可以在 **Paddle + MKLDNN、ONNX CPU、ONNX CUDA GPU** 三种后端之间切换，
  用户不必换插件目录。

### v0.9（ONNX CPU 默认）

- 问题：没有 NVIDIA 显卡的用户仍需要一个稳定、容易部署的本地识别路径；Paddle / MKLDNN
  的环境差异也容易让初次安装变得复杂。
- 修复：把 ONNX Runtime CPU 作为默认选择，补齐 `setup.bat` 的安装检查，并提供纯净部署包与
  含 V6 模型的 CPU 懒人包。没有 GPU 也能解压后直接开始识别。

**宿主补丁用法（v1.3）**：完整发布包已内嵌，无需再 patch。若把插件装进
**官方原版 Umi**，先退出软件，再运行：

```bat
patches\umi-host\apply_host_patches.bat
REM 或
patches\umi-host\apply_host_patches.bat "D:\path\to\Umi-OCR"
```

会备份原文件后覆盖批量任务、表格和 PDF 文字层所需的 19 个宿主文件。说明见
[`patches/umi-host/README.md`](./patches/umi-host/README.md)。

更细的提交说明见本仓库 `git log`。

---

## 支持

- **Issue / 讨论**：请在本仓库提 issue：<https://github.com/chapterv/umi-paddle-neoengine/issues>  
  尽量附上：Umi 与插件路径、推理引擎选项、是否 GPU、`engine_stderr.log` 末尾、样张是否可公开  
- **相关项目**  
  - Umi-OCR：<https://github.com/hiroi-sora/Umi-OCR>  
  - 云端 AI OCR 插件（精度对照参考）：<https://github.com/EatWorld/UmiOCR-AI-OCR-Plugin>  

---

## 友情链接

- **[LINUX DO](https://linux.do)** — 真诚、友善、充满活力的技术社区，本项目认可并推荐

---

## 开源协议

- 本插件代码遵循与 **Umi-OCR / 本仓库声明** 一致的开源约定（未单独附加时按 MIT 类惯例与上游兼容使用）。  
- **PaddlePaddle / PaddleOCR / ONNX Runtime** 等依赖遵循其各自许可证。  
- **模型权重** 使用须遵守百度飞桨 / PaddleOCR 模型协议；请勿将未授权商用场景误用。  

---

**感谢使用 umi-paddle-neoengine。** 本地升级到 PP-OCRv6，可选 GPU，继续离线、免费、可批量。
