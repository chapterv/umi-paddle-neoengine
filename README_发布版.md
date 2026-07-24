# Local-Ocr · Umi-OCR 新引擎（PP-OCRv6）发布包

**当前版本：1.3**（与源码仓 `VERSION` 对齐）

把 Umi-OCR v2.1.5 内置的 PP-OCRv3 引擎，升级为官方最新 **PaddleOCR 3.x** 路线
（推荐 **PP-OCRv6 medium** + **ONNX Runtime**，可在设置里切 v5/v4 与 Paddle/MKLDNN）。

> **v1.3**：修复竖排 PDF 的方向、旋转页、斜框和漏标点问题；双层 PDF
> 优先保证 Chrome/Edge 的局部高亮与单栏连续复制，单层 PDF 增加
> 自动/横排/竖排选择。
> 锁定 **`paddlepaddle==3.2.1`**（3.3.x oneDNN 在 Windows/CPU 下易崩，勿擅自升级）。  
> 公开源码：<https://github.com/chapterv/umi-paddle-neoengine>

---

## 更新日志

### v1.3（2026-07-24）

- **竖排双层可搜索 PDF**
  - 自动识别窄高竖排文字框，每个 OCR 栏写入一个标准嵌入 CJK 连续 run；
    Chrome/Edge 可局部高亮，复制单栏时文字连续。
  - 支持 0°/90°/180°/270° 页面和轻微倾斜四点框；布局阶段明确给出的
    `end` 标点不会再被输出层丢弃。
  - Chrome/Edge 跨多个独立竖栏拖选时仍按几何左→右且可能插入换行。
    Tagged PDF、`ActualText` 与 W2 单流均无法同时保留局部高亮并重排剪贴板，
    因此本版明确保留该阅读器限制。严格语序请选单层 `统一横排` 或同时导出 TXT。
  - 原始扫描图像保持不变；横排文档继续使用原有写入方式。
- **单层纯文字 PDF**
  - 勾选 `text.pdf 单层纯文本文档` 后，可设置
    `自动（保留 OCR 排版）`、`统一横排`或`统一竖排`。
  - 自动为默认值，允许同一页横竖混排；竖排使用标准 CJK 纵向字体指标，
    汉字保持直立且文本完整。该设置不影响双层 PDF。
- **竖排句号与性能**
  - 只在原图存在唯一高置信环形连通域且尺寸、位置、视觉 cell、低墨量均可信时
    恢复 `。`，不根据换行或语义猜标点；可用
    `UMI_OCR_VERTICAL_PUNCTUATION_RECOVERY=0` 紧急关闭。
  - P1 只有在 `table.csv + P1` 的明确请求下才启动；普通截图、预览、文档 OCR
    固定走 `task=ocr`。GPU 短文本 ABBA 稳态实测额外中位耗时为 +1.717%。
- **兼容验证**
  - PDF 行为 12/12、C3 5/5、P1 路由 4/4、引擎契约 8/8、追踪 5/5、
    宿主补丁 4/4、P0 表格 65/65。

### v1.2（2026-07-23）

- **P0 几何表格**
  - 新增 `表格-几何网格` 排版解析和 `table.csv 结构表格(Excel)` 输出。
  - 结构 CSV 会在普通排版前直接使用原始 OCR 坐标建表，避免坐标丢失后猜列。
  - 输出 UTF-8 BOM CSV，可用 Excel 直接打开。
- **P1 可选结构模型**
  - 接入 PaddleOCR `TableRecognitionPipelineV2`，复用当前 PP-OCRv6 识别结果。
  - 支持 rowspan/colspan 合并表头；模型失败自动回退 P0。
  - 默认关闭；额外依赖和约 955 MB 模型仅在用户选择时安装。
- **安装与升级**
  - `setup.bat` 新增第 4 步 P1 选择，直接回车默认 `N`。
  - 新增 `install_table_models.bat`，自动选择与 `run.cmd` 相同的
    `.venv_gpu → .venv` 环境，可用于老版本补装。

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

**宿主补丁（可选）**：zip 解压即用无需 patch。官方 Umi + 只拷插件时：

```bat
patches\umi-host\apply_host_patches.bat "你的\Umi-OCR路径"
```

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

## 当前发布包（v1.3）

输出目录：同级 **`umi-paddle-neoengine-release/`**

| 包 | 引擎默认 | 包含 | 解压后 | 适合 |
|----|------|------|--------|------|
| **umi-paddle-neoengine-deploy-v1.3.zip** | ONNX CPU | 源码 + setup.bat（**不含** venv / 模型） | 双击 `setup.bat` → 打开 Umi-OCR | 有网、最小包 |
| **umi-paddle-neoengine-ONNX-V6-CPU-v1.3.zip** | ONNX CPU | 含精简 `.venv` + V6 ONNX 等模型 | 直接双击 `Umi-OCR\Umi-OCR.exe` | 懒人 / 离线 |

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
2. 双击根目录 **`setup.bat`**
   - 第 1 步：选模型范围（完整 / 最小可用 / 多语言）
   - 第 2 步：选推理后端 —— `[1]` 纯 GPU（onnxruntime-gpu）/ `[2]` 纯 CPU（默认）/ `[3]` 全安装
   - 选 GPU 时第 3 步：选 CUDA 版本 —— `[1]` 1.27.0 + CUDA13 / `[2]` 1.26.0 + CUDA12.9（默认，RTX 30/40 系）
   - 自动建 `.venv_gpu` 虚拟环境并 `pip install paddlepaddle==3.2.1 paddleocr==3.7.0 onnxruntime-gpu[cuda,cudnn]==1.26.0`
   - 自动预下载所选模型（约 1 分钟，需联网）
   - 第 4 步询问是否安装 **P1 表格结构模型**；直接回车默认 `N`
   - 输入 `Y` 会安装可选依赖并预下载约 **955 MB** 的表格模型
   - 全程约 2~3 分钟
   - ⚠️ 选中 GPU 但本机 CUDA 不可用，引擎会明确标 `[⚠回退CPU]` 并打 WARN，功能仍正常（仅无 GPU 提速）
3. 双击 `Umi-OCR\Umi-OCR.exe` 打开主程序
4. 「全局设置 → 文字识别」里选 **PaddleOCR·PP-OCRv6/v4（新引擎）**
5. 拖入图片开始识别（首次切到 v6 会自动下载 v6 模型，仅需一次）

已有可用环境、不想重跑完整 setup：

```bat
install_table_models.bat
```

该入口会复用 `run.cmd` 的环境选择顺序，优先 `.venv_gpu`、再回退 `.venv`。
`install_table_models.bat --check` 只检查，不安装；`--deps-only` 只装依赖，
模型在首次开启 P1 时下载。

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

## 表格识别与结构 CSV

### P0 几何方式：无需安装额外模型

适合边框清楚、行列较规整的表格，速度和普通 OCR 接近。

1. 打开 **批量识图**或**批量文档**页面。
2. 右侧切换到 **设置**。
3. 展开 **保存文件类型**，勾选
   **`table.csv 结构表格(Excel)`**。
4. 直接开始任务即可。即使排版方案保持原选项，系统也会在普通排版前按
   原始 OCR 坐标自动建立二维表。
5. 如果还要让结果文本区以 TSV 表格形式预览，在
   **OCR文本后处理 → 排版解析方案**选择
   **`表格-几何网格`**。

输出位置遵循页面中的 **保存到** 设置，文件名以 `_table.csv` 结尾。CSV 使用
UTF-8 BOM，可直接双击用 Excel 打开。

### P1 模型方式：复杂表与合并单元格

首次使用前，通过 `setup.bat` 第 4 步输入 `Y`，或双击
`install_table_models.bat`。安装完成后：

1. **完全退出并重启 Umi-OCR**。
2. 左侧进入 **全局设置**。
3. 打开 **文字识别**，确认 **当前接口**为本项目的
   PaddleOCR 本地新引擎。
4. 找到开关 **`表格结构模型（P1·可选）`**并开启。
5. 点击该组顶部的绿色按钮 **`应用修改`**；看到
   “文字识别接口应用成功”才算生效。
6. 回到批量识图/批量文档，仍然勾选
   **`table.csv 结构表格(Excel)`**后开始任务。

关闭 P1 开关并再次点击 **应用修改**，即可恢复纯 P0/普通 OCR；更改开关时
若已有任务运行，需先等待任务结束或点击 **强制终止任务**。

### P0 与 P1 的区别

| 项目 | P0 几何方式 | P1 结构模型 |
|------|-------------|-------------|
| 安装 | 默认已有，无额外依赖 | 可选安装，约 955 MB 模型 |
| 原理 | OCR 文本框坐标聚类成行列 | SLANeXt/单元格检测 + OCR 填格 |
| 速度/内存 | 轻量，接近普通 OCR | 冷启动和内存明显更高 |
| 规整有线表 | 推荐 | 可用 |
| 无线表/复杂表 | 能力有限 | 通常更好 |
| 合并单元格 | CSV 中只能展开/近似 | 保留 HTML rowspan/colspan，再展开到 CSV |
| 失败处理 | 无模型失败点 | 自动回退 P0，普通 OCR 不报错 |

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
├─ install_table_models.bat       # P1 表结构模型可选安装/升级
├─ Umi-OCR/
│  ├─ Umi-OCR.exe               # 主程序（官方 v2.1.5）
│  └─ UmiOCR-data/
│     └─ plugins/
│        └─ win_x64_PaddleOCR_Py/   # ← 本项目的全部改动都在这一个文件夹（两引擎共用）
│           ├─ engine.py             # Python 引擎 worker（MKLDNN 修复 + ONNX 旁路）
│           ├─ run.cmd              # 入口：调 .venv/python engine.py
│           ├─ PPOCR_*.py          # Umi-OCR 插件契约
│           ├─ requirements.txt     # paddlepaddle==3.2.1 / paddleocr==3.7.0 / onnxruntime-gpu[cuda,cudnn]==1.26.0（默认 CUDA 12.9）/ 1.27.0（CUDA 13）
│           ├─ requirements-table.txt # P1 可选依赖（默认基础安装不装）
│           ├─ table_structure.py    # P1 HTML/合并单元格 → 通用 table 协议
│           ├─ download_table_models.py # P1 依赖检查与模型预下载
│           ├─ models/configs.txt  # 语言下拉框配置（必须保留）
│           ├─ .venv_gpu/        # [仅懒人版] Python 环境（GPU 用 onnxruntime-gpu）
│           └─ paddlex/           # [仅懒人版] 官方模型缓存（真实权重）
└─ README_发布版.md
```

> 想自己改引擎代码？所有改动只在 `plugins/win_x64_PaddleOCR_Py/` 一个目录，
> Umi-OCR 主程序零改动。
