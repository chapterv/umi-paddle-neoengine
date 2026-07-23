# engine.py —— Route B 的 Python 引擎 worker（v4 · MKLDNN修复 + 性能优化 + GPU/ONNX 回退）
#
# 协议：启动打印 "OCR init completed."；逐行读 stdin JSON → stdout JSON。
# 运行环境：run.cmd → .venv/python → paddlepaddle **3.2.1** + paddleocr 3.7.0
#
# ⚠️ paddlepaddle 必须是 3.2.1！3.3.x 的 PIR+oneDNN 兼容性 bug 会导致
#    MKLDNN 推理崩溃 (ConvertPirAttribute2RuntimeAttribute [pir::ArrayAttribute])
#    详见 https://github.com/PaddlePaddle/paddle/issues/77340
#
# v4 变更记录（2026-07-17 21:49 最终迭代）：
#   A. 性能修复：use_cls 默认 False / limit_side_len 默认 1920 / 每请求后 gc / stderr 计时日志
#   B. 精度修复：自动白边补丁（pad=50，解决 PP-OCRv6 边缘丢首字）
#   C. 约束修正：MKLDNN 默认开启（paddle 3.2.1 已修复 oneDNN 崩溃）
#
# 2026-07-18 GPU 增量（Route A · ONNX + CUDAExecutionProvider）：
#   - --engine onnxruntime-gpu → providers=["CUDAExecutionProvider","CPUExecutionProvider"]
#     优先 CUDA，不可用时 **自动回退 CPU**（CUDA 缺失 / cuDNN 未装 / 某 op 不支持均安全降级）。
#   - --engine onnxruntime → 纯 CPU。
#   - --fallback_chain（§7）→ 可配置版本回退链，覆盖默认 V6→V5→V4。
# 2026-07-19 合并为单一插件（CPU/GPU 合一）：本文件同时服务 paddle(MKLDNN) / ONNX-CPU /
#   ONNX-CUDA 三种后端，由 PPOCR_config.py 的「推理引擎」下拉框选择；requirements.txt 同时
#   装入 onnxruntime-gpu[cuda,cudnn]，故 CUDA 无需系统级 CUDA Toolkit 即可用，不可用时自动回退。
import os
import sys

# 2026-07-19 CUDA DLL 搜索路径修复（仅 Windows，import onnxruntime 之前执行）：
# onnxruntime-gpu 的 [cuda] extras 会把 nvidia-cublas / cuda-runtime / cudnn / cufft / curand / nvrtc
#   的 cu12 DLL 自动装进 site-packages/nvidia/<lib>/bin/；但 ORT 的 CUDA EP 在加载
#   onnxruntime_providers_cuda.dll 时，由 Windows 加载器**静态依赖** cublasLt64_12.dll，
#   而 ORT 的预载只覆盖了 runtime/cudnn/cufft/curand/nvrtc，**漏了 cublasLt**
#   → 报 "depends on cublasLt64_12.dll which is missing" (Error 126)，CUDA EP 起不来。
# 解决（双保险，本段在 import onnxruntime 之前运行）：
#   ① os.add_dll_directory(nvidia/*/bin)（Py3.8+，加入进程 DLL 搜索路径）；
#   ② 把同目录并进 os.environ["PATH"] —— **实测①单独在 ORT 内部 LoadLibrary 时不生效，
#     ②（PATH）才是 Windows 上让 CUDA EP 真正加载 cublasLt 的关键**；两者都加最稳。
#   注意：site-packages 路径必须用 site.getsitepackages() 取，
#   不能用 os.path.dirname(os.__file__)（那是 Lib/ 不是 Lib/site-packages，glob 会落空）。
if os.name == "nt":
    try:
        import glob as _glob
        import site as _site
        _roots = []
        try:
            _roots += _site.getsitepackages()
        except Exception:
            pass
        _roots.append(os.path.join(os.path.dirname(os.__file__), "site-packages"))
        for _sp in _roots:
            for _d in _glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass
                try:
                    _p = os.environ.get("PATH", "")
                    if _d not in _p:
                        os.environ["PATH"] = _d + os.pathsep + _p
                except Exception:
                    pass
    except Exception:
        pass
import io
import json
import base64
import argparse
import importlib.util
import time
import gc
import tempfile

import numpy as np
import cv2
from PIL import Image

# ── 模型自包含：把 paddle 官方模型缓存重定向到插件自己的 paddlex/ 目录 ──
# 必须在 import paddleocr 之前设置 PADDLE_PDX_CACHE_HOME（paddlex 读取此变量）。
# 这样：① 首次识别自动下载模型到此目录（无需联网后手动搬运）；
#      ② 懒人版直接预置模型到此目录，解压即用。
# （GPU 插件可通过目录 junction 共享 CPU 插件的 paddlex/，避免模型存两份。）
#
# ⚠️ Windows + 非 ASCII 路径（如「发布包」）：
#   Paddle 原生推理 C++ 层 IsFileExists 对 Unicode 路径常失败，报
#   Cannot open file ...\inference.json（文件其实在、Python 也能 open）。
#   ONNXRuntime 一般没事；engine=paddle 会必崩。
#   修复：PADDLE_PDX_CACHE_HOME 尽量改成 8.3 短路径（纯 ASCII），指向同一目录。
def _win_short_path(path: str) -> str:
    if os.name != "nt" or not path:
        return path
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    try:
        import ctypes
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        # 路径必须已存在，短名才解析得出来
        if not os.path.exists(path):
            return path
        buf = ctypes.create_unicode_buffer(4096)
        n = GetShortPathNameW(path, buf, 4096)
        if n and buf.value:
            try:
                buf.value.encode("ascii")
                return buf.value
            except UnicodeEncodeError:
                return buf.value
    except Exception:
        pass
    return path


_HERE = os.path.dirname(os.path.abspath(__file__))
_PADDLEX_HOME = os.path.join(_HERE, "paddlex")
os.makedirs(_PADDLEX_HOME, exist_ok=True)
_PADDLEX_HOME = _win_short_path(os.path.abspath(_PADDLEX_HOME))
os.environ["PADDLE_PDX_CACHE_HOME"] = _PADDLEX_HOME
if _PADDLEX_HOME != os.path.join(_HERE, "paddlex"):
    try:
        sys.stderr.write(
            f"[engine] PADDLE_PDX_CACHE_HOME 使用 ASCII 短路径（规避中文路径）：{_PADDLEX_HOME}\n"
        )
    except Exception:
        pass

from paddleocr import PaddleOCR
from table_structure import attach_table_result, structure_output_to_table

# ── 白边补丁常量（解决边缘丢字问题）───────────────────────────────
PAD = 50  # 四边各留 50px 白色边距（实测 margin≥50 不丢字）

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--models_path", default=None)
parser.add_argument("--config_path", default="models/config_chinese.txt")
parser.add_argument("--cls", default=None)
parser.add_argument("--use_angle_cls", default=None)
parser.add_argument("--limit_side_len", default="1920")
parser.add_argument("--cpu_threads", default="6")
parser.add_argument("--enable_mkldnn", default="True")  # paddlepaddle 3.2.1 已修复 PIR bug
parser.add_argument("--ocr_version", default=None)  # 覆盖默认版本优先级
parser.add_argument(
    "--table_structure",
    default="False",
    help="是否允许宿主发送 task=table；默认关闭，模型首次使用时才加载",
)
# ── 文档预处理开关（默认开，对齐 PaddleOCR 3.x 原生行为；GUI 可关）──
parser.add_argument("--use_doc_orientation", default=None,
                    help="文档方向纠正（整图旋转）：None=默认开 / true/false")
parser.add_argument("--use_doc_unwarping", default=None,
                    help="文档去扭曲（曲面矫正/UVDoc）：None=默认开 / true/false")
parser.add_argument("--use_textline_orientation", default=None,
                    help="纠正文本方向（逐行 0/180°）：None=默认开 / true/false")
# ── 推理后端 ───────────────────────────────────────────────────────
# engine=None/paddle/mkldnn/空 → Paddle 原生后端（走 MKLDNN，CPU）；
# engine=onnxruntime → ONNX Runtime CPU 旁路；
# engine=onnxruntime-gpu → ONNX Runtime CUDA GPU（不可用时自动回退 CPU）。
parser.add_argument("--engine", default=None,
                    help="推理引擎：None=paddle(MKLDNN) / onnxruntime / onnxruntime-gpu")
parser.add_argument("--engine_config", default=None,
                    help='JSON 字符串，如 {"providers":["CUDAExecutionProvider","CPUExecutionProvider"]}')
# ── 可配置版本回退链（§7 fallback_chain）──────────────────────────
# 逗号分隔，覆盖默认 V6→V5→V4。如 "PP-OCRv6,PP-OCRv4" 去掉 V5。
parser.add_argument("--fallback_chain", default=None,
                    help='版本回退链，逗号分隔，如 "PP-OCRv6,PP-OCRv5,PP-OCRv4"；覆盖默认优先级')
args = parser.parse_args()


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def parse_lang_key(config_path):
    name = os.path.basename(str(config_path))
    if name.endswith(".txt"):
        name = name[:-4]
    if name.startswith("config_"):
        name = name[len("config_"):]
    return name


LANG_MAP = {
    "chinese": "ch",
    "en": "en",
    "english": "en",
    "chinese_cht": "chinese_cht",
    "japan": "japan",
    "japanese": "japan",
    "korean": "korean",
    "cyrillic": "ru",
    "ru": "ru",
}
raw_lang = parse_lang_key(args.config_path)
lang = LANG_MAP.get(raw_lang, raw_lang)

use_cls = False                          # 对齐 PPOCR_config 默认值
if args.use_textline_orientation is not None:
    use_cls = to_bool(args.use_textline_orientation)
elif args.cls is not None:
    use_cls = to_bool(args.cls)
elif args.use_angle_cls is not None:
    use_cls = to_bool(args.use_angle_cls)

cpu_threads = int(args.cpu_threads)
limit_side_len = int(args.limit_side_len)
enable_mkldnn = to_bool(args.enable_mkldnn)  # paddlepaddle 3.2.1 已修复 PIR+oneDNN bug，默认开启加速
TABLE_STRUCTURE_ENABLED = to_bool(args.table_structure)

# ── 引擎后端（ONNX Runtime）────────────────────────────────────────
# onnxruntime / onnxruntime-gpu 都走 ONNX Runtime 后端；
# 区别仅在于 provider：gpu 版优先 CUDA，不可用时自动回退 CPU。
ENGINE = (getattr(args, "engine", None) or "").strip().lower()
IS_ONNX = (ENGINE in ("onnxruntime", "onnxruntime-gpu"))
_engine_config_raw = getattr(args, "engine_config", None)
ENGINE_CONFIG = None
if _engine_config_raw:
    try:
        ENGINE_CONFIG = json.loads(_engine_config_raw)
    except Exception:
        sys.stderr.write(f"[engine] engine_config 解析失败，忽略：{_engine_config_raw!r}\n")
        ENGINE_CONFIG = None
if IS_ONNX and ENGINE_CONFIG is None:
    # ── GPU + 可选择 fallback 规则 ──
    # onnxruntime-gpu：优先 CUDAExecutionProvider，不可用时自动回退 CPUExecutionProvider
    #   （CUDA 缺失 / cuDNN 未装 / 某 op 在 CUDA 不被支持时，ORT 自动降级 CPU，插件照常工作）。
    # onnxruntime：纯 CPU。
    if ENGINE == "onnxruntime-gpu":
        ENGINE_CONFIG = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    else:
        ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
# onnxruntime 后端不使用 MKLDNN；保持 enable_mkldnn=False 避免混淆
if IS_ONNX:
    enable_mkldnn = False

# ── 后端标签（供日志/输出标注 cpu/gpu + 模型版本）──
ENGINE_LABEL = "?"   # 如 gpu(onnx-cuda) / cpu(onnx) / cpu(paddle)
DEVICE = "?"         # gpu / cpu
ORT_VERSION = None   # onnxruntime 版本
CUDA_VER = None      # 检测到的 CUDA 版本（如 12.x）
LOADED_VER = None    # build_ocr 实际加载成功的 ocr_version（回退后可能与请求不同）
USED_MKLDNN = None  # build_ocr 实际生效的 mkldnn（paddle 后端；回退后可能与请求不同）


def _detect_cuda_version():
    """从已装的 nvidia-cuda-runtime-cu1x 包探测 CUDA 大版本。"""
    try:
        import importlib.metadata as _md
        for pkg, maj in (("nvidia-cuda-runtime-cu13", "13"),
                         ("nvidia-cuda-runtime-cu12", "12")):
            try:
                v = _md.version(pkg)
                return f"{maj}.x ({v})"
            except Exception:
                continue
    except Exception:
        pass
    return None


def _cuda_gpu_present():
    """用 CUDA 驱动层确认本机**真有**可用 GPU 设备（不只是 onnxruntime 列了 CUDA EP）。

    onnxruntime 的 get_available_providers() 会列出 CUDAExecutionProvider
    即使它实际跑不起来（CUDA DLL 半残 / 驱动不匹配 / 本机无 GPU），
    这正是「选了 gpu 却跑 cpu、还标 gpu」的掩耳盗铃根因——
    本机实测 get_available_providers/get_device 都会**间歇性**把残 CUDA 列为可用。
    这里下沉到 CUDA 驱动层（nvcuda.dll / cuDeviceGetCount）做硬确认：
    能加载 nvcuda 且设备数>0 才算真有 GPU。"""
    try:
        import ctypes
        lib = ctypes.CDLL("nvcuda.dll")
        # ⚠️ 必须先 cuInit(0) 初始化 CUDA 驱动，否则 cuDeviceGetCount
        #   直接返回 CUDA_ERROR_NOT_INITIALIZED(=3) → 误判「无 GPU」，
        #   导致本机真有 GPU 也被判成无 → 推理永远回退 CPU（掩耳盗铃反向坑）。
        if lib.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
        return count.value > 0
    except Exception as e:
        sys.stderr.write(f"[engine][WARN] CUDA 驱动层检测失败（视为无 GPU）：{e}\n")
        return False


def _resolve_engine_label():
    """返回 (ENGINE_LABEL, DEVICE, ORT_VERSION, CUDA_VER)。"""
    if not IS_ONNX:
        return "cpu(paddle)", "cpu", None, None
    try:
        import onnxruntime as ort
        ov = ort.__version__
        cv = _detect_cuda_version()
        using_cuda = bool(ENGINE_CONFIG) and "CUDAExecutionProvider" in ENGINE_CONFIG.get("providers", [])
        if using_cuda:
            return "gpu(onnx-cuda)", "gpu", ov, cv
        return "cpu(onnx)", "cpu", ov, cv
    except Exception:
        return "cpu(onnx?)", "cpu", None, None

# ── 文档预处理开关（由本引擎自管「方向纠正 + 去扭曲」并精确逆映射坐标）──
# 这两个开关直接来自 GUI / PPOCR_umi（恒为显式 True/False）。
# 重要背景：paddleocr 3.x 在开启 use_doc_unwarping / use_doc_orientation 时，
#   内部会对图像做几何变换后再检测，但**不再把检测框逆映射回原图坐标系**
#   （PaddleOCR 2.x / 旧版 V3 引擎的 DocUnwarping 是会做这一步的）。
#   这正是一开这些功能、检测框就偏移几十像素、与原图不对齐的根因。
# 修复：build_ocr 里把这两项恒设为 False（交还 paddleocr 内部预处理，避免重复），
#   改由下方 preprocess_doc() 自行完成「垫白边 → 方向纠正 → 透视矫正」，
#   并对检测框做**精确逆变换**映射回原图坐标。→ 功能保持开启、框完美对齐，
#   且不依赖 paddleocr 3.x 内部被砍掉的逆映射实现。
_use_doc_ori = getattr(args, "use_doc_orientation", None)
use_doc_orientation = to_bool(_use_doc_ori) if _use_doc_ori is not None else False
_use_doc_unwarp = getattr(args, "use_doc_unwarping", None)
use_doc_unwarping = to_bool(_use_doc_unwarp) if _use_doc_unwarp is not None else False

# ═════════════════════════════════════════════════════════════════
# 版本优先级构建（核心决策：用户选择第一，回退链服从）
# ═════════════════════════════════════════════════════════════════
#
# 决策规则（铁律）：
#   ① 用户选择的模型版本（--ocr_version）永远是第一位尝试目标；
#   ② 回退链（fallback_chain / fallback_1~3）仅决定「剩余候选」的顺序；
#   ③ V6 不支持的语言（韩/俄）→ **直接删除 V6**（无论用户是否显式选 V6）。
#      理由：V6 无该语言模型。paddle 后端会抛 ValueError（R4 已 catch → 回退），
#      但 onnx 后端会**静默加载中文 rec 模型**→输出 "?" 垃圾（不抛错，回退链失效）。
#      故统一在版本列表里删除 V6，不让它进入尝试矩阵。
#
# 示例：
#   用户选V6；回退链 v5,v4 → 实际 [V6, V5, V4]        ← 无冲突
#   用户选V6；回退链 v5,v4,v6 → 实际 [V6, V5, V4]    ← 去重
#   用户选V5；回退链 v6,v5,v4 → 实际 [V5, V6, V4]    ← 用户优先
#   用户选V5；回退链 v4,v6     → 实际 [V5, V4, V6]    ← 回退链补充
#   用户选V6 + 韩文           → 实际 [V5, V4]        ← V6删除(防乱码/卡死)
#

# ── 第1步：确定用户显式选择的版本（永远第0位）────────────
_ocr_version_arg = getattr(args, "ocr_version", None) or ""
_user_ver = _ocr_version_arg.strip() or "PP-OCRv6"   # 用户选择，默认 V6

# ── 第2步：取回退链（GUI 的 fallback_1/2/3 拼成）───
_DEFAULT_CHAIN = ["PP-OCRv6", "PP-OCRv5", "PP-OCRv4"]
_fc = getattr(args, "fallback_chain", None) or ""
if _fc.strip():
    _chain = [v.strip() for v in _fc.split(",") if v.strip()]
else:
    _chain = list(_DEFAULT_CHAIN)

# ── 第3步：合并——用户选择放首位，回退链去重追加在后 ───
VERSION_PRIORITY = [_user_ver]
for _v in _chain:
    if _v not in VERSION_PRIORITY:
        VERSION_PRIORITY.append(_v)

sys.stderr.write(
    f"[engine] 版本决策：用户选={_user_ver} | 回退链={_chain} "
    f"→ 实际尝试顺序={VERSION_PRIORITY}\n"
)

# ── 第4步：V6 不支持语言的安全调整（韩/俄）──────────────
# V6 无 Korean/Russian 模型。不同后端行为：
#   - paddle: V6+韩/俄 → ValueError 立即抛（R4 已 catch → 回退 V5）✅
#   - onnx  : V6+韩/俄 → **静默加载中文 rec 模型** → 输出 "?" 垃圾
#             （不抛错，回退链失效，R4 catch 抓不到）❌
# 因此对不支持的语言，**无论用户是否显式选 V6、无论后端**，
# 都从优先列表中**删除** V6，直接走 V5/V4。
# 用户显式选 V6 也无效——V6 无该语言模型，强行加载只会乱码（ONNX）或抛错（paddle）。
# 下方 _init_ocr_with_timeout 的 catch 仍保留作 paddle 后端的安全网。
_V6_UNSUPPORTED_LANGS = {"korean", "ru"}
if lang in _V6_UNSUPPORTED_LANGS and "PP-OCRv6" in VERSION_PRIORITY:
    _before = list(VERSION_PRIORITY)
    VERSION_PRIORITY = [v for v in VERSION_PRIORITY if v != "PP-OCRv6"]
    _why = "用户显式选了 V6" if _user_ver == "PP-OCRv6" else "V6 来自回退链"
    sys.stderr.write(
        f"[engine][WARN] 语言 {lang} 在 PP-OCRv6 无模型（{_why}）。"
        f"已从版本优先列表删除 V6：{_before} → {VERSION_PRIORITY}\n"
    )


def _init_ocr_with_timeout(ver, mk, timeout_sec=90):
    """在独立线程中初始化 PaddleOCR，带超时保护。

    背景：PaddleOCR 构造函数在某些「版本×语言」组合下会**永久阻塞**
    （如 PP-OCRv6 + korean/ru：V6 无该语言模型，PaddleX 内部卡死
    在模型下载/加载等待中，不返回、不抛异常）。
    若不设超时，except 永远抓不到 → for 循环卡死 → 回退链失效 →
    用户看到「不报错也不出字、一直卡着」。

    返回：(ocr实例, None) 成功 或 (None, 异常) 失败（含超时TimeoutError）。
    """
    import concurrent.futures

    def _do_init():
        return PaddleOCR(
            device="cpu",
            lang=lang,
            ocr_version=ver,
            use_textline_orientation=use_cls,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=mk,
            cpu_threads=cpu_threads,
            text_det_limit_side_len=limit_side_len,
            text_det_limit_type="max",
            **({"engine": "onnxruntime", "engine_config": ENGINE_CONFIG}
               if IS_ONNX else {}),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_init)
        try:
            ocr = future.result(timeout=timeout_sec)
            return ocr, None
        except concurrent.futures.TimeoutError:
            future.cancel()  # 放弃该线程（进程退出时自动回收）
            return None, TimeoutError(
                f"PaddleOCR({ver}, lang={lang}) 初始化超时 "
                f"({timeout_sec}s)，可能该版本不支持此语言，已跳过"
            )
        except Exception as e:
            # ⚠️ R4 根因修复（2026-07-20）：
            # 旧代码只 catch TimeoutError，不 catch 其他异常。
            # 实测 V6+韩文/俄文会**立即抛 ValueError**（"No models are available
            # for lang='korean' and ocr_version='PP-OCRv6'"，约 2s），而非铁律 #2
            # 所述「永久阻塞」。该 ValueError 穿透 build_ocr -> main -> engine.py
            # 整体崩溃（exit 1）-> 不打印 "OCR init completed." -> 回退链失效。
            # catch 后返回 (None, e)，让 build_ocr continue 跳下一版本（V5）。
            return None, e


def build_ocr():
    """按 VERSION_PRIORITY 依次尝试加载（每个尝试有超时保护），首个成功即用。

    超时保护：每个版本/后端组合的 PaddleOCR 构造函数最多等待 90 秒。
    超时自动判定为失败，继续尝试下一个版本——解决 V6+韩文/俄文
    「永久阻塞导致回退链失效」的问题。
    """
    # ⚠️ ENGINE_CONFIG 必须声明为 global：本函数内对它的重新赋值
    #   （ONNX 后端按可用性收敛为 CUDA 或 CPU）若不声明 global，会变成「局部变量」，
    #   模块级 ENGINE_CONFIG（import 时按 onnxruntime-gpu 默认设成了 CUDA+CPU）
    #   永远不会被更新 → _resolve_engine_label 读到陈旧 CUDA 值 →
    #   实际已回退 CPU，却仍标 gpu(cuda)（掩耳盗铃 bug，2026-07-19 修复）。
    global ENGINE_CONFIG
    # 报告 ONNX 后端实际可用的 provider，并按可用性**最终决定**用哪个 provider：
    #   - onnxruntime-gpu 且 CUDAExecutionProvider 可用 → 优先 CUDA，CPU 兜底；
    #   - onnxruntime-gpu 但 CUDA 不可用（没装 onnxruntime-gpu / 驱动不匹配 / 缺 DLL）
    #     → 自动降级为纯 CPUExecutionProvider，**不再硬崩**，功能正常但无 GPU 提速；
    #   - onnxruntime → 纯 CPU。
    # paddlex 在请求到不可用的 provider 时会直接抛异常，所以必须由这里先按可用性收敛。
    if IS_ONNX:
        try:
            import onnxruntime as ort
        except ImportError as e:
            # 部署包常见根因：只装了 paddleocr，推理后端 pip 中断/失败，
            # 或仅装了残缺 venv。GUI 默认 engine=onnxruntime 时会连跪 V6/V5/V4。
            raise RuntimeError(
                "engine=%s 但当前 Python 环境未安装 onnxruntime（import 失败：%s）。\n"
                "请在插件目录重新运行项目根 setup.bat，并完成「推理后端」安装：\n"
                "  · 纯 CPU：选项 [2] → 安装 onnxruntime 到 .venv\n"
                "  · GPU：选项 [1]/[3] → 安装 onnxruntime-gpu 到 .venv_gpu\n"
                "安装完成后可用:  .venv\\Scripts\\python -c \"import onnxruntime; print(onnxruntime.__version__)\"\n"
                "或:              .venv_gpu\\Scripts\\python -c \"import onnxruntime; print(onnxruntime.__version__)\""
                % (ENGINE, e)
            ) from e
        try:
            av = ort.get_available_providers()
            sys.stderr.write(f"[engine] ORT 可用 providers: {av}\n")
            if ENGINE == "onnxruntime-gpu":
                # 双保险：provider 列表「且」驱动层真有 GPU 设备，才启用 CUDA。
                # 单看 get_available_providers 会被残 CUDA 骗（间歇性列出 CUDA 却跑不起来）。
                if "CUDAExecutionProvider" in av and _cuda_gpu_present():
                    ENGINE_CONFIG = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
                    sys.stderr.write("[engine] ✅ CUDAExecutionProvider 可用，启用 GPU 推理。\n")
                else:
                    ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
                    sys.stderr.write(
                        "[engine][WARN] 本机未检测到可用 CUDA GPU"
                        "（onnxruntime-gpu 未安装，或 CUDA DLL/驱动不可用，或 cuDeviceGetCount==0）。\n"
                        "         已自动回退 CPUExecutionProvider：功能正常，但无 GPU 提速。\n"
                        "         若需 GPU：在该插件 .venv 安装 onnxruntime-gpu[cuda,cudnn]==1.26.0"
                        "（无需系统级 CUDA Toolkit）。\n"
                    )
            else:
                ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
                sys.stderr.write("[engine] 纯 CPU ONNX 推理。\n")
        except RuntimeError:
            raise
        except Exception as e:
            sys.stderr.write(f"[engine][WARN] 检查 providers 失败，回退 CPU：{e}\n")
            ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
    # 解析最终后端标签（cpu/gpu + onnxruntime/CUDA 版本），供日志与输出标注
    global ENGINE_LABEL, DEVICE, ORT_VERSION, CUDA_VER
    ENGINE_LABEL, DEVICE, ORT_VERSION, CUDA_VER = _resolve_engine_label()
    sys.stderr.write(
        f"[engine] 后端标签：{ENGINE_LABEL} / onnxruntime={ORT_VERSION}"
        f"{('/CUDA ' + CUDA_VER) if CUDA_VER else ''}\n"
    )
    # ── 构造「版本 × mkldnn」尝试矩阵 ──
    # ONNX 后端：mkldnn 无意义，仅按版本尝试。
    # Paddle 后端：若用户开了 mkldnn，先试「开」；失败则**自动回退「关」**
    #   （mkldnn 不可用 / oneDNN 崩溃 / 缺 DLL 等），不再硬崩。
    #   这正是「CPU 模块之间不 fallback」的修复点：paddle+mkldnn 失败
    #   → 自动改用 paddle 无 mkldnn，功能照常。
    attempts = []
    for ver in VERSION_PRIORITY:
        if IS_ONNX:
            attempts.append((ver, False))
        else:
            if enable_mkldnn:
                attempts.append((ver, True))
                attempts.append((ver, False))   # mkldnn 回退档
            else:
                attempts.append((ver, False))
    last_err = None
    for ver, mk in attempts:
        ocr, err = _init_ocr_with_timeout(ver, mk, timeout_sec=90)
        if err:
            last_err = err
            sys.stderr.write(f"[engine][WARN] {ver} mkldnn={mk} 初始化失败：{err}\n")
            continue
        # 初始化成功
        global LOADED_VER, USED_MKLDNN
        LOADED_VER = ver
        USED_MKLDNN = mk
        sys.stderr.write(
            f"[engine] 已加载：{ver} / lang={lang} / cls={use_cls} "
            f"/ mkldnn={mk} / limit={limit_side_len}\n"
        )
        if (not IS_ONNX) and enable_mkldnn and (mk is False):
            sys.stderr.write(
                "[engine][WARN] 请求的 mkldnn 初始化失败，"
                "已自动回退 paddle 无 mkldnn（CPU）。功能正常。\n"
            )
        return ocr, ver
    tried = "、".join(f"{v}(mkldnn={m})" for v, m in attempts)
    raise RuntimeError(
        f"无法初始化 PaddleOCR（已尝试 {tried}，lang={lang}）：{last_err}"
    )


def load_image(req):
    """从请求取图像路径；返回 None(跳过) / 路径字符串 / 'unknown'。"""
    if "image_path" in req:
        p = req["image_path"]
        if p == "clipboard":
            return None
        return p
    elif "image_base64" in req:
        raw = base64.b64decode(req["image_base64"])
        fd, tmp = tempfile.mkstemp(suffix=".jpg")     # .jpg 更小且截图源通常就是 JPEG
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return tmp
    return "unknown"


def _pad_image(path):
    """给图像加 PAD px 白色边距，返回新临时文件路径。
    调用方负责用完后删除该临时文件。
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return path  # 打开失败则原样传回
    w, h = img.size
    padded = Image.new("RGB", (w + 2 * PAD, h + 2 * PAD), (255, 255, 255))
    padded.paste(img, (PAD, PAD))
    fd, tmp = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        padded.save(f, format="PNG")
    return tmp


def _unpad_box(box, pad=PAD):
    """将补丁后的全局坐标还原到原图坐标系（减去 PAD）。"""
    return [[int(round(x - pad)), int(round(y - pad))] for x, y in box]


def convert(result, was_padded=True):
    """paddleocr 3.x list[dict] → Umi-OCR {code,data:[{box,score,text}],backend}.

    was_padded: 输入图像是否经过 _pad_image 处理（UVDoc 模式为 False，跳过 unpad）。
    """
    if not result or len(result) == 0:
        return {"code": 100, "data": [], "backend": _backend_tag()}
    item = result[0]
    polys = item.get("rec_polys") or item.get("dt_polys") or []
    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or []
    data = []
    for poly, text, score in zip(polys, texts, scores):
        if was_padded:
            box = _unpad_box(poly)                       # ← 还原坐标
        else:
            box = [[int(round(x)), int(round(y))] for x, y in poly]
        data.append({"box": box, "score": float(score), "text": str(text)})
    return {"code": 100, "data": data, "backend": _backend_tag()}


# ======================================================================
# 文档预处理（方向纠正 + 去扭曲）—— 自管坐标逆映射
# ----------------------------------------------------------------------
# 背景：paddleocr 3.x 开启 use_doc_unwarping / use_doc_orientation 时，
#   内部对图像做几何变换后再检测，但**不再把检测框逆映射回原图坐标系**
#   （PaddleOCR 2.x / 旧版 V3 引擎的 DocUnwarping 是会做这一步的）。
#   这正是一开这些功能、检测框就偏移几十像素、与原图不对齐的根因。
# 修复：build_ocr 把这两项恒设为 False（禁用 paddleocr 内部预处理，避免重复变换），
#   改由下面的 preprocess_doc() 自行完成「垫白边 → 方向纠正 → 透视矫正」，
#   并用**精确逆变换**（单应矩阵 H⁻¹、旋转逆阵、去白边）把检测框
#   映射回原图坐标。→ 功能保持开启、框完美对齐，且不依赖 paddleocr 内部被砍掉的逆映射。
# 鲁棒性：任一步失败/不适用时对应逆变换退化为恒等，保证框不偏移、不崩溃。
# ======================================================================

_DOC_PREP_PIPE = None   # None=未建；False=加载失败哨兵

def _load_doc_ori():
    """惰性构建 doc_preprocessor 管线（仅用于取「方向角度」）。

    为何用整条管线而非直接 jit 加载 PP-LCNet 分类模型：
      paddleocr 3.x 把方向分类模型打包成**推理图**，
      `paddle.jit.load` 直接加载会报 KeyError('forward')（没有可直接调用的 Layer）。
      而 doc_preprocessor 管线由 paddlex 自己负责加载，能正确返回 angle。
      ⚠️ 该管线内部也含 UVDoc 去扭曲——但我们**只用 angle**，
         绝不取它的 output_img（那会重新引入「框不回原图」的偏移 bug）。
         去扭曲由本引擎的 cv2 透视矫正 + 精确逆映射完成。
    """
    global _DOC_PREP_PIPE
    if _DOC_PREP_PIPE is not None:
        return _DOC_PREP_PIPE
    try:
        from paddlex import create_pipeline
        _DOC_PREP_PIPE = create_pipeline(pipeline="doc_preprocessor")
        sys.stderr.write("[preproc] doc_preprocessor 管线（取方向角度）已加载。\n")
    except Exception as e:
        sys.stderr.write(f"[preproc][WARN] doc_preprocessor 管线加载失败，方向纠正将跳过：{e}\n")
        _DOC_PREP_PIPE = False
    return _DOC_PREP_PIPE


def _classify_orientation(img_bgr):
    """返回文档方向角度 0/90/180/270；管线缺失或失败返回 0。"""
    pipe = _load_doc_ori()
    if not pipe:
        return 0
    try:
        r = next(iter(pipe.predict(img_bgr)))
        return int(r.get("angle", 0))
    except Exception as e:
        sys.stderr.write(f"[preproc][WARN] 方向分类失败，按 0° 处理：{e}\n")
        return 0


def _order_corners(pts):
    """把 4 个顶点重排为 左上/右上/右下/左下。"""
    pts = pts[np.argsort(pts[:, 1])]
    top, bot = pts[:2], pts[2:]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype="float32")


def _detect_doc_corners(img_bgr):
    """检测文档四边形顶点（垫白边后的图像坐标）；检测不到返回 None。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    H, W = img_bgr.shape[:2]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype("float32")
        x, y, w, h = cv2.boundingRect(pts)
        # 只接受「像文档」的四边形：面积足够大、长宽比合理，
        # 避免对截图/UI 误矫正（那种应走恒等、不产生偏移）。
        if w * h < 0.5 * H * W:
            continue
        if not (0.2 < (w / h) < 5.0):
            continue
        return pts
    return None


def preprocess_doc(img_bgr, do_ori, do_unwarp, pad=PAD):
    """自管文档预处理，返回 (pre_img, inv_fn)。

    pre_img : 送 OCR 的图（已垫白边 + 可能的旋转/透视矫正）。
    inv_fn  : 把 pre_img 坐标系的 4 点框映射回【原图】坐标系
              （逆透视 → 逆旋转 → 去白边），功能开着时框完美对齐原图。
    任一步失败/不适用时对应逆变换退化为恒等：框不偏移、不崩溃。
    """
    H0, W0 = img_bgr.shape[:2]
    # 1) 垫白边（避免 V6 边缘丢字；也为后续变换留出边距）
    padded = np.full((H0 + 2 * pad, W0 + 2 * pad, 3), 255, dtype="uint8")
    padded[pad:pad + H0, pad:pad + W0] = img_bgr
    cur = padded
    M_inv = None   # 逆旋转矩阵（None=未旋转）
    H_inv = None   # 逆透视矩阵（None=未矫正）
    # 2) 方向纠正（0/90/180/270）
    if do_ori:
        angle = _classify_orientation(cur)
        if angle not in (0, None):
            h, w = cur.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(angle), 1.0)
            cur = cv2.warpAffine(cur, M, (w, h), borderValue=(255, 255, 255))
            M_inv = cv2.invertAffineTransform(M)
    # 3) 去扭曲（文档四边形透视矫正）
    if do_unwarp:
        corners = _detect_doc_corners(cur)
        if corners is not None:
            src = _order_corners(corners)
            w_top = np.linalg.norm(src[1] - src[0])
            w_bot = np.linalg.norm(src[2] - src[3])
            h_l = np.linalg.norm(src[3] - src[0])
            h_r = np.linalg.norm(src[2] - src[1])
            dw = max(int(round((w_top + w_bot) / 2)), 1)
            dh = max(int(round((h_l + h_r) / 2)), 1)
            dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype="float32")
            Hm = cv2.getPerspectiveTransform(src, dst)
            cur = cv2.warpPerspective(cur, Hm, (dw, dh), borderValue=(255, 255, 255))
            H_inv = np.linalg.inv(Hm)
    pre_img = cur

    def inv_fn(box):
        out = []
        for (x, y) in box:
            if H_inv is not None:
                p = np.array([x, y, 1.0], dtype="float64")
                q = H_inv @ p
                x, y = q[0] / q[2], q[1] / q[2]
            if M_inv is not None:
                p = np.array([x, y, 1.0], dtype="float64")
                q = M_inv @ p
                x, y = q[0], q[1]
            out.append([int(round(x)) - pad, int(round(y)) - pad])
        return out

    return pre_img, inv_fn


def _is_fallback():
    """是否发生了「回退」（实际加载的 ≠ 用户选择的）。

    判定基准：_user_ver（用户通过 GUI --ocr_version 显式选择的版本）。
    只有实际加载的版本与用户选择不同时才算回退（加 | fb）。

    示例：
      用户选 V6 → 加载 V6 = 不回退 → 无 | fb
      用户选 V6 → V6 不可用、加载 V5 = 回退 → | fb
      用户选 V5 → 加载 V5 = 不回退 → 无 | fb
    """
    if LOADED_VER and LOADED_VER != _user_ver:
        return True          # 版本 ≠ 用户选择
    if ENGINE == "onnxruntime-gpu" and DEVICE == "cpu":
        return True          # 请求 GPU 却跑 CPU
    if (not IS_ONNX) and enable_mkldnn and USED_MKLDNN is False:
        return True          # 请求 mkldnn 却关了
    return False


def _backend_tag():
    """实际使用的推理模块（不是用户「请求」的，而是 build_ocr 收敛后真正跑的）。

    正常情况直接写实际模块，如 cpu(onnx) v6 / gpu(cuda) v6；
    仅当发生回退时，右侧追加「 | fb」短标记（绝不写「GPU回退」之类大字）。
    """
    ver = LOADED_VER or getattr(args, "ocr_version", None) or "PP-OCRv6"
    ver_short = ver.replace("PP-OCRv", "") if "PP-OCRv" in ver else "6"
    if DEVICE == "gpu":
        tag = "gpu(cuda)"                 # 真·GPU（CUDA 实际可用）
    elif IS_ONNX:
        tag = "cpu(onnx)"                # onnxruntime / onnxruntime-gpu 落到 onnx-CPU
    else:
        mk = USED_MKLDNN if USED_MKLDNN is not None else enable_mkldnn
        tag = "cpu(paddle+mkldnn)" if mk else "cpu(paddle)"
    tag += f" v{ver_short}"
    fb = _is_fallback()
    if fb:
        tag += " | fb"
    # 诊断日志（stderr → 引擎可见；帮助定位标签异常）
    sys.stderr.write(
        f"[tag] {tag}  (LOADED_VER={LOADED_VER}, USER_CHOICE={_user_ver}, "
        f"DEVICE={DEVICE}, IS_ONNX={IS_ONNX}, fallback={fb})\n"
    )
    return tag


def get_rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        return -1


def run_ocr_safe(ocr, img):
    """推理期间把 OS 级 stdout(fd1) 重定向到 stderr(fd2)。

    背景 bug（用户实测 V6+MKL 报 904）：
      paddle/oneDNN 在 C 层（phi.dll）向 stdout 打印诊断信息
      `[ReduceMeanCheckIfOneDNNSupport]`，污染了结果 JSON 协议行，
      导致宿主 Umi-OCR 把首行当 JSON 解析 → 反序列化失败(904)。
      V4 的图不触发该 op 故正常；V6 检测模型命中该 op 故必崩。
    解法：推理期间把 fd1 指向 fd2（stderr，本就是我们的日志流），
    推理结束恢复 fd1 再写唯一 JSON，保证 stdout 干净。
    """
    saved_fd = os.dup(1)
    sys.stdout.flush()
    os.dup2(2, 1)  # stdout -> stderr
    try:
        result = ocr.ocr(img)
    finally:
        os.dup2(saved_fd, 1)  # 恢复 stdout
        os.close(saved_fd)
    return result


_TABLE_PIPELINE = None
_GEOMETRY_BUILDER = None


def _table_pipeline_kwargs():
    if IS_ONNX:
        return {"engine": "onnxruntime", "engine_config": ENGINE_CONFIG}
    return {"engine": "paddle"}


def _get_table_pipeline():
    """首次 task=table 时才加载可选模型，默认 OCR 启动零额外开销。"""
    global _TABLE_PIPELINE
    if _TABLE_PIPELINE is None:
        from paddleocr import TableRecognitionPipelineV2

        _TABLE_PIPELINE = TableRecognitionPipelineV2(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            # 复用本进程已经完成的 PP-OCRv6；避免官方表管线再加载一套
            # PP-OCRv4（其 server_det 当前没有 ONNX 官方包）。
            use_ocr_model=False,
            **_table_pipeline_kwargs(),
        )
    return _TABLE_PIPELINE


def _read_image_array(img):
    if isinstance(img, np.ndarray):
        return img
    arr = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"table image read failed: {img}")
    return arr


def _overall_ocr_result(img, blocks):
    """Umi textBlocks → PaddleX table pipeline 的 overall_ocr_res。"""
    polys = []
    boxes = []
    texts = []
    scores = []
    for block in blocks or []:
        poly = block.get("box") or []
        if len(poly) != 4:
            continue
        points = [[int(round(p[0])), int(round(p[1]))] for p in poly]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        polys.append(points)
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        texts.append(str(block.get("text") or ""))
        scores.append(float(block.get("score") or 0.0))
    image_array = _read_image_array(img)
    poly_array = np.asarray(polys, dtype=np.int32)
    return {
        "input_path": img if isinstance(img, str) else None,
        "page_index": None,
        "doc_preprocessor_res": {"output_img": image_array},
        "dt_polys": poly_array,
        "rec_polys": poly_array.copy(),
        "rec_boxes": np.asarray(boxes, dtype=np.int32),
        "rec_texts": texts,
        "rec_scores": scores,
    }


def run_table_structure(img, ocr_blocks):
    """运行 PaddleOCR 表结构模型并返回通用 table；stdout 始终保持 JSON-only。"""
    saved_fd = os.dup(1)
    sys.stdout.flush()
    os.dup2(2, 1)
    try:
        pipeline = _get_table_pipeline()
        output = list(
            pipeline.predict(
                input=img,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_ocr_model=False,
                overall_ocr_res=_overall_ocr_result(img, ocr_blocks),
                use_table_orientation_classify=False,
            )
        )
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    table = structure_output_to_table(output)
    if not table:
        raise RuntimeError("table structure pipeline returned no table")
    return table


def _get_geometry_builder():
    """复用宿主 P0 纯函数；按文件加载，避免触发 Umi 包级初始化。"""
    global _GEOMETRY_BUILDER
    if _GEOMETRY_BUILDER is not None:
        return _GEOMETRY_BUILDER
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "py_src",
            "ocr",
            "tbpu",
            "parser_tools",
            "table_grid.py",
        )
    )
    spec = importlib.util.spec_from_file_location("umi_table_grid_fallback", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load geometry fallback: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GEOMETRY_BUILDER = module.build_table
    return _GEOMETRY_BUILDER


def main():
    t_init = time.time()
    ocr, ver = build_ocr()

    sys.stdout.write("OCR init completed.\n")
    sys.stdout.flush()

    try:
        import datetime
        _log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_active.log")
        with open(_log, "a", encoding="utf-8") as f:
            _fb = " | fb" if _is_fallback() else ""
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
                    f"init_ok ver={ver} lang={lang} cls={use_cls} "
                    f"engine={ENGINE} backend={ENGINE_LABEL} device={DEVICE} "
                    f"onnxruntime={ORT_VERSION}{('/CUDA '+CUDA_VER) if CUDA_VER else ''}{_fb} "
                    f"limit={limit_side_len} threads={cpu_threads} "
                    f"init_sec={time.time()-t_init:.1f}\n")
    except Exception:
        pass

    req_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        task = req.get("task") or "ocr"

        req_count += 1
        t0 = time.time()
        tmp_padded = None          # 需要清理的补丁临时文件
        tmp_base64 = None          # 需要清理的 base64 临时文件
        try:
            img = load_image(req)
            if img is None:
                out = {"code": 100, "data": []}
            elif isinstance(img, str) and img == "unknown":
                out = {"code": 901, "data": "未知指令"}
            else:
                # 判断是否为 base64 产生的临时文件（需后续删除）
                is_b64_tmp = (isinstance(img, str) and
                              os.path.dirname(img) in (tempfile.gettempdir(),))
                if is_b64_tmp:
                    tmp_base64 = img

                # ── 文档预处理（方向纠正 / 去扭曲）由本引擎自管 ──
                # 开启任一项时，preprocess_doc 完成「垫白边 → 方向纠正 → 透视矫正」，
                # 并返回 inv_fn 把检测框精确逆映射回【原图】坐标系
                # （功能保持开启、框完美对齐；paddleocr 内部预处理已在 build_ocr 禁用）。
                # 两项都关时走原 V6 边缘丢字补丁路径（垫白边，convert 内 unpad）。
                _do_pre = use_doc_orientation or use_doc_unwarping
                tmp_padded = None
                inv_fn = None
                if _do_pre:
                    try:
                        # ⚠️ preprocess_doc 期望 BGR numpy 数组，但 img 是文件路径字符串。
                        # 旧代码直接传路径 -> 'str' object has no attribute 'shape' ->
                        # except 吞掉 -> 回退 run_img=img（无 padding 原图）。
                        # 后果：GUI 默认 use_doc_*=True 时，连基础白边（_pad_image）都被
                        # 跳过 -> V6/V5 边缘丢字。修复：先 cv2.imread 转成 ndarray。
                        if isinstance(img, str):
                            # cv2.imread 在 Windows 不支持非 ASCII 路径（如中文文件名
                            # 样章-韩语段落.png），实测返回 None。用 np.fromfile + imdecode 绕过。
                            _img_arr = cv2.imdecode(
                                np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                        else:
                            _img_arr = img
                        if _img_arr is None:
                            raise RuntimeError(f"图像读取失败：{img}")
                        run_img, inv_fn = preprocess_doc(_img_arr, use_doc_orientation, use_doc_unwarping)
                    except Exception as e:
                        sys.stderr.write(
                            f"[preproc][WARN] 预处理异常，回退无预处理 OCR：{e}\n"
                        )
                        run_img, inv_fn = img, None
                else:
                    tmp_padded = _pad_image(img)   # V6 边缘丢字补丁
                    run_img = tmp_padded

                try:
                    result = run_ocr_safe(ocr, run_img)
                    out = convert(result, was_padded=(not _do_pre))
                    # 预处理开启：把检测框从预处理图坐标系逆映射回原图
                    if _do_pre and inv_fn is not None:
                        for _d in out.get("data", []):
                            try:
                                _d["box"] = inv_fn(_d["box"])
                            except Exception as _e:
                                sys.stderr.write(f"[preproc][WARN] 框逆映射失败跳过：{_e}\n")
                    if task == "table" and TABLE_STRUCTURE_ENABLED:
                        structure_table = None
                        structure_error = ""
                        try:
                            structure_table = run_table_structure(
                                img, out.get("data") or []
                            )
                        except Exception as table_exc:
                            structure_error = (
                                f"{type(table_exc).__name__}: {table_exc}"
                            )
                            sys.stderr.write(
                                "[table][WARN] 结构模型失败，回退几何网格："
                                f"{structure_error}\n"
                            )
                        geometry_builder = None
                        if structure_table is None:
                            try:
                                geometry_builder = _get_geometry_builder()
                            except Exception as geometry_exc:
                                structure_error += (
                                    " | geometry loader "
                                    f"{type(geometry_exc).__name__}: {geometry_exc}"
                                )
                        out = attach_table_result(
                            out,
                            structure_table,
                            geometry_builder=geometry_builder,
                            structure_error=structure_error,
                        )
                finally:
                    # 仅清理 V6 补丁产生的临时文件
                    if tmp_padded and tmp_padded != img and os.path.exists(tmp_padded):
                        try:
                            os.remove(tmp_padded)
                        except Exception:
                            pass
                        tmp_padded = None
        except Exception as e:
            out = {"code": 902, "data": f"识别失败：{type(e).__name__}: {e}"}

        dt = time.time() - t0
        rss = get_rss_mb()
        _data = out.get("data") if isinstance(out, dict) else None
        _scores = [d.get("score", 0) for d in _data if isinstance(d, dict)] if isinstance(_data, list) else []
        _avg_conf = (sum(_scores) / len(_scores)) if _scores else 0.0
        sys.stderr.write(
            f"[engine] #{req_count} {dt:.2f}s rss={rss}MB "
            f"backend={ENGINE_LABEL} ver={ver} conf={_avg_conf:.2f} "
            f"task={task} table={out.get('table', {}).get('source', '-')} "
            f"out_code={out.get('code')} n_text={len(_scores)}\n"
        )

        sys.stdout.write(json.dumps(out, ensure_ascii=True) + "\n")
        sys.stdout.flush()

        # 清理 base64 临时文件
        if tmp_base64 and os.path.exists(tmp_base64):
            try:
                os.remove(tmp_base64)
            except Exception:
                pass

        gc.collect()


if __name__ == "__main__":
    main()
