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
_HERE = os.path.dirname(os.path.abspath(__file__))
_PADDLEX_HOME = os.path.join(_HERE, "paddlex")
os.makedirs(_PADDLEX_HOME, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = _PADDLEX_HOME

from paddleocr import PaddleOCR

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

# ocr_version：优先加载的版本（默认 v6 medium 精度最高；v5 回退用于韩/俄等 v6 未覆盖语言；v4 兜底最快）
_ocr_version_arg = getattr(args, "ocr_version", None)
DEFAULT_VERSION = "PP-OCRv6"
VERSION_PRIORITY = [DEFAULT_VERSION, "PP-OCRv5", "PP-OCRv4"]
if _ocr_version_arg and _ocr_version_arg.strip():
    VERSION_PRIORITY = [_ocr_version_arg.strip()] + [v for v in VERSION_PRIORITY if v != _ocr_version_arg.strip()]
# ── 可配置版本回退链（§7 fallback_chain）──
# 覆盖默认优先级；如 "PP-OCRv6,PP-OCRv5,PP-OCRv4"。
# 用户可去掉 V5/V4 或调整顺序，无需改代码。GPU/ONNX 不可用时按此链回退到下一版本。
_fc = getattr(args, "fallback_chain", None)
if _fc and _fc.strip():
    VERSION_PRIORITY = [v.strip() for v in _fc.split(",") if v.strip()]
    sys.stderr.write(f"[engine] 使用自定义回退链：{VERSION_PRIORITY}\n")


def build_ocr():
    """按 VERSION_PRIORITY 依次尝试加载，首个成功即用。"""
    # 报告 ONNX 后端实际可用的 provider，并按可用性**最终决定**用哪个 provider：
    #   - onnxruntime-gpu 且 CUDAExecutionProvider 可用 → 优先 CUDA，CPU 兜底；
    #   - onnxruntime-gpu 但 CUDA 不可用（没装 onnxruntime-gpu / 驱动不匹配 / 缺 DLL）
    #     → 自动降级为纯 CPUExecutionProvider，**不再硬崩**，功能正常但无 GPU 提速；
    #   - onnxruntime → 纯 CPU。
    # paddlex 在请求到不可用的 provider 时会直接抛异常，所以必须由这里先按可用性收敛。
    if IS_ONNX:
        try:
            import onnxruntime as ort
            av = ort.get_available_providers()
            sys.stderr.write(f"[engine] ORT 可用 providers: {av}\n")
            if ENGINE == "onnxruntime-gpu":
                if "CUDAExecutionProvider" in av:
                    ENGINE_CONFIG = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
                    sys.stderr.write("[engine] ✅ CUDAExecutionProvider 可用，启用 GPU 推理。\n")
                else:
                    ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
                    sys.stderr.write(
                        "[engine][WARN] 本机未检测到 CUDAExecutionProvider"
                        "（onnxruntime-gpu 未安装，或 CUDA DLL/驱动不可用）。\n"
                        "         已自动回退 CPUExecutionProvider：功能正常，但无 GPU 提速。\n"
                        "         若需 GPU：在该插件 .venv 安装 onnxruntime-gpu[cuda,cudnn]==1.26.0"
                        "（无需系统级 CUDA Toolkit）。\n"
                    )
            else:
                ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
                sys.stderr.write("[engine] 纯 CPU ONNX 推理。\n")
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
    last_err = None
    for ver in VERSION_PRIORITY:
        try:
            ocr = PaddleOCR(
                device="cpu",
                lang=lang,
                ocr_version=ver,
                use_textline_orientation=use_cls,
                # 方向纠正 / 去扭曲 交还本引擎自管（preprocess_doc）：
                # 这两个开关 paddleocr 3.x 内部开启时会做几何变换，
                # 但**不再把检测框逆映射回原图**，导致框偏移、不对齐。
                # 故此处恒为 False，由 preprocess_doc() 完成变换并精确逆映射坐标。
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=enable_mkldnn,
                cpu_threads=cpu_threads,
                text_det_limit_side_len=limit_side_len,
                text_det_limit_type="max",  # 只缩小不放大：细图按原生尺寸跑，避免被放大 40+ 倍导致极慢
                **({"engine": "onnxruntime", "engine_config": ENGINE_CONFIG}
                   if IS_ONNX else {}),
            )
            sys.stderr.write(
                f"[engine] 已加载：{ver} / lang={lang} / cls={use_cls} "
                f"/ limit={limit_side_len}\n"
            )
            return ocr, ver
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"无法初始化 PaddleOCR（已尝试 {VERSION_PRIORITY}，lang={lang}）：{last_err}"
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


def _backend_tag():
    """生成简洁后端标签供 UI 显示，如 'gpu v6' / 'cpu(paddle) v6'。"""
    dev = DEVICE                    # "gpu" / "cpu"
    eng = ENGINE                    # "paddle" / "onnxruntime" / "onnxruntime-gpu"
    if eng == "onnxruntime-gpu":
        eng_short = "cuda"
    elif eng == "onnxruntime":
        eng_short = "onnx"
    else:
        eng_short = "paddle"
    # args.ocr_version 真实启动由 PPOCR_umi 下发；此处做 None 防御
    ov = getattr(args, "ocr_version", None) or "PP-OCRv6"
    ver_short = ov.replace("PP-OCRv", "") if "PP-OCRv" in ov else "6"
    return f"{dev}({eng_short}) v{ver_short}"


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


def main():
    t_init = time.time()
    ocr, ver = build_ocr()

    sys.stdout.write("OCR init completed.\n")
    sys.stdout.flush()

    try:
        import datetime
        _log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_active.log")
        with open(_log, "a", encoding="utf-8") as f:
            _fb = " [⚠回退CPU]" if (ENGINE == "onnxruntime-gpu" and DEVICE == "cpu") else ""
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
                        run_img, inv_fn = preprocess_doc(img, use_doc_orientation, use_doc_unwarping)
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
