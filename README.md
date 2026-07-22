# umi-paddle-neoengine

**Umi-OCR 本地 PP-OCRv6 引擎插件**（Route B：Python 插件调用官方 PaddleOCR 3.x）

[![Version](https://img.shields.io/badge/version-1.1-orange)](./VERSION)
[![Umi-OCR](https://img.shields.io/badge/Umi--OCR-v2.1.5-blue)](https://github.com/hiroi-sora/Umi-OCR)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.7-green)](https://github.com/PaddlePaddle/PaddleOCR)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/github-chapterv%2Fumi--paddle--neoengine-black)](https://github.com/chapterv/umi-paddle-neoengine)

面向 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) 的**本地离线**新引擎插件：在**不改主程序**的前提下，把识别能力从内置老旧 PP-OCRv3，升级到官方 **PaddleOCR 3.x（PP-OCRv6 / v5 / v4）**，并支持 **ONNX CPU / ONNX CUDA GPU / Paddle+MKLDNN** 三种推理后端。

- **当前源码版本**：**1.1**（见仓库根目录 [`VERSION`](./VERSION)，Git 标签 `v1.1`；开发以 **`master`** 为准）  
- **本仓库（源码）**：<https://github.com/chapterv/umi-paddle-neoengine>  
- **完整发布包**（含 Umi 主程序 + `setup.bat`）：同级目录 **`umi-paddle-neoengine-release/`**（zip **不进**本 git 仓库）  
  - `umi-paddle-neoengine-deploy.zip` — 纯净部署（需 `setup.bat`）  
  - `umi-paddle-neoengine-ONNX-V6-CPU.zip` — ONNX V6 CPU 懒人包  
- **宿主补丁（主程序 py_src 修复）**：[`patches/umi-host/`](./patches/umi-host/)  
  - 完整 zip 已内嵌；若只装插件、主程序仍是官方原版，请运行  
    [`patches/umi-host/apply_host_patches.bat`](./patches/umi-host/apply_host_patches.bat)  
    （可拖入 `Umi-OCR` 目录；会先备份再覆盖 5 个文件）

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

一句话：**Umi 壳子不动，本地识别引擎换代到 PP-OCRv6，并可选 GPU。**

---

## 对比识别效果

> 下列素材均在 `images/` 目录。性能样章原图：[`样章-性能测试.png`](images/样章-性能测试.png)（约 1718×1188，密集中文表格）。

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

### 残缺 / 低质样张

**原图：**

![样章-残缺识别](images/样章-残缺识别.png)

**本插件 v6 识别结果：**

![残缺识别1](images/残缺识别1.png)

低质、残缺印刷在 v6 下仍能稳住主体文字，明显优于老 v3 引擎常见「大片空白 / 乱码」。

---

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
| **部署脚本** | 根目录 `setup.bat`：模型范围 + 推理后端（CPU/GPU）两段式安装 |
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
   - **`umi-paddle-neoengine-deploy.zip`**：小包，需联网 `setup.bat`  
   - **`umi-paddle-neoengine-ONNX-V6-CPU.zip`**：含 `.venv` + V6 ONNX 模型，默认 ONNX CPU  
2. 解压到**尽量纯英文路径**（如 `C:\Local-Ocr\`；中文路径下 paddle 原生更易出问题，ONNX 相对稳）  
3. 纯净包：双击根目录 **`setup.bat`**  
   - 模型范围：默认「最小可用 = 中文 V6 ONNX」即可  
   - 推理后端：默认 **[2] 纯 CPU**；有 N 卡选 **[1] GPU**，第 3 步默认 **1.26 + CUDA12.9**  
4. 运行 `Umi-OCR\Umi-OCR.exe`  
5. **全局设置 → 文字识别**：引擎选本插件（PaddleOCR 新引擎）  
6. 推理引擎建议：  
   - 无 GPU / 要稳 → **ONNX Runtime CPU**  
   - 有 N 卡要快 → **ONNX Runtime CUDA GPU**（并视需要关闭「文档去扭曲」以接近 ~2s）  

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

**如何确认 GPU 生效：**  
识别结果耗时旁或日志中出现 `gpu(cuda)` / `gpu(onnx-cuda)`；若出现回退 CPU 的 WARN，检查是否装了 `onnxruntime-gpu` 且驱动正常。

**速度对照（同性能测试图，本机示例）：**

| 条件 | 约热身耗时 |
|------|------------|
| ONNX GPU + 预处理关 | **~1.5–1.9s** |
| ONNX GPU + 预处理全开 | **~3.2s** |
| ONNX CPU / paddle CPU | **十余秒** 量级 |

---

## 版本修订记录

版本号写在仓库根目录 **`VERSION`**（当前 **`1.1`**），发布时打 Git 标签 **`v1.1`**。

| 版本 / 阶段 | 说明 |
|-------------|------|
| **1.1（当前）** | 批量文档：空白页 `median([])` 防护；Mission `forceRecover` / worker epoch；stop 杀引擎可再启；单页 OCR 约 180s 超时。发布包：`deploy` + `ONNX-V6-CPU`。宿主补丁见 [`patches/umi-host/`](./patches/umi-host/)，仅插件场景用 [`apply_host_patches.bat`](./patches/umi-host/apply_host_patches.bat) 一键覆盖 Umi `py_src` |
| **1.0** | 默认引擎 `onnxruntime`；纯净包 + ONNX V6 CPU 懒人包；setup 校验 ort；里程碑标签 `1.0` |
| **GPU 路线** | `onnxruntime-gpu` + CUDA EP；DLL PATH 修复；缺 CUDA 自动回退 |
| **MKLDNN 稳定** | 锁定 `paddlepaddle==3.2.1`，修复 3.3.x PIR/oneDNN 崩溃与 904 协议污染 |
| **多语言与回退** | 韩/俄等无 v6 rec 时回退 v5；语言码映射（如 ru） |
| **路径加固** | Windows 非 ASCII 路径下 Paddle 缓存改 8.3 短路径 |
| **部署** | `setup.bat` 两段式；修复 echo `>` 误生成垃圾文件；打包排除 bench 与误产物 |

**宿主补丁用法（v1.1）**：完整发布包已内嵌，无需再 patch。若把插件装进**官方原版 Umi**，先退出软件，再运行：

```bat
patches\umi-host\apply_host_patches.bat
REM 或
patches\umi-host\apply_host_patches.bat "D:\path\to\Umi-OCR"
```

会备份原文件后覆盖 `mission*.py` / `BatchDOC.py` / `line_preprocessing.py`。说明见 [`patches/umi-host/README.md`](./patches/umi-host/README.md)。

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
