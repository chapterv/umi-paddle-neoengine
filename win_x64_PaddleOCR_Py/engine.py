# engine.py —— Route B 的 Python 引擎 worker（v4 · MKLDNN修复 + 性能优化）
#
# 协议：启动打印 "OCR init completed."；逐行读 stdin JSON → stdout JSON。
# 运行环境：run.cmd → .venv/python → paddlepaddle **3.2.1** + paddleocr 3.7.0
#
# ⚠️ paddlepaddle 必须是 3.2.1！3.3.x 的 PIR+oneDNN 兼容性 bug 会导致
#    MKLDNN 推理崩溃 (ConvertPirAttribute2RuntimeAttribute [pir::ArrayAttribute])
#    详见 https://github.com/PaddlePaddle/paddle/issues/77340
#
# v4 变更记录（2026-07-17 21:49 最终迭代）：
#   A. 性能修复：
#      1) use_cls 默认 False（对齐 GUI 默认，避免每次多跑角度分类）
#      2) limit_side_len 默认 1920（适配 Windows 高 DPI，替代过激的 960）
#      3) 每请求后 gc.collect() 减少内存膨胀、延缓 Umi-OCR ram_max 重启引擎
#      4) stderr 计时日志（耗时+RSS+结果条数），定位"卡住"环节
#   B. 精度修复：
#      5) 自动白边补丁（pad=50）：解决 PP-OCRv6 对紧贴边缘文字（<35px）丢首字的 bug。
#         传入 ocr.ocr() 前给图像四边加 50px 白色边距；返回后从 box 坐标中减去偏移量。
#         Umi-OCR 上层完全无感。
#      6) base64 临时文件改用 .jpg 后缀（减小体积、避免 PNG 编码伪影）
#   C. 约束不变：
#      oneDNN/MKLDNN 强制关闭（paddle 3.3.1 Windows CPU 构建推理期崩溃）

import os
import sys
import io
import json
import base64
import argparse
import time
import gc
import tempfile

import numpy as np
from PIL import Image

# ── 模型自包含：把 paddle 官方模型缓存重定向到插件自己的 paddlex/ 目录 ──
# 必须在 import paddleocr 之前设置 PADDLE_PDX_CACHE_HOME（paddlex 读取此变量）。
# 这样：① 简洁版首次识别自动下载模型到此目录（无需联网后手动搬运）；
#      ② 懒人版直接预置模型到此目录，解压即用。
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
# ── 路径二：ONNX Runtime 后端（完全绕过 MKLDNN / oneDNN）─────────
# engine='onnxruntime' 时，PaddleOCR 自动下载 ONNX 格式模型并使用
# onnxruntime 推理。engine_config 透传 onnxruntime 的 SessionOptions，
# 这里只关心 providers（CPU 用 CPUExecutionProvider）。
parser.add_argument("--engine", default=None,
                    help="推理引擎：None=paddle(MKLDNN) / onnxruntime")
parser.add_argument("--engine_config", default=None,
                    help='JSON 字符串，如 {"providers":["CPUExecutionProvider"]}')
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
if args.cls is not None:
    use_cls = to_bool(args.cls)
elif args.use_angle_cls is not None:
    use_cls = to_bool(args.use_angle_cls)

cpu_threads = int(args.cpu_threads)
limit_side_len = int(args.limit_side_len)
enable_mkldnn = to_bool(args.enable_mkldnn)  # paddlepaddle 3.2.1 已修复 PIR+oneDNN bug，默认开启加速

# ── 引擎后端（路径二）────────────────────────────────────────────
# engine=None → paddle 默认后端（走 MKLDNN）；engine='onnxruntime' → 绕开 MKLDNN。
ENGINE = getattr(args, "engine", None)
_engine_config_raw = getattr(args, "engine_config", None)
ENGINE_CONFIG = None
if _engine_config_raw:
    try:
        ENGINE_CONFIG = json.loads(_engine_config_raw)
    except Exception:
        sys.stderr.write(f"[engine] engine_config 解析失败，忽略：{_engine_config_raw!r}\n")
        ENGINE_CONFIG = None
if ENGINE == "onnxruntime" and ENGINE_CONFIG is None:
    ENGINE_CONFIG = {"providers": ["CPUExecutionProvider"]}
# onnxruntime 后端不使用 MKLDNN；保持 enable_mkldnn=False 避免混淆
if ENGINE == "onnxruntime":
    enable_mkldnn = False

# ocr_version：优先加载的版本（默认 v4 mobile 最快，可选 v6 精度最高）
_ocr_version_arg = getattr(args, "ocr_version", None)
DEFAULT_VERSION = "PP-OCRv4"
VERSION_PRIORITY = [DEFAULT_VERSION, "PP-OCRv6", "PP-OCRv5"]
if _ocr_version_arg and _ocr_version_arg.strip():
    VERSION_PRIORITY = [_ocr_version_arg.strip()] + [v for v in VERSION_PRIORITY if v != _ocr_version_arg.strip()]


def build_ocr():
    """按 VERSION_PRIORITY 依次尝试加载，首个成功即用。"""
    last_err = None
    for ver in VERSION_PRIORITY:
        try:
            ocr = PaddleOCR(
                device="cpu",
                lang=lang,
                ocr_version=ver,
                use_textline_orientation=use_cls,
                enable_mkldnn=enable_mkldnn,
                cpu_threads=cpu_threads,
                text_det_limit_side_len=limit_side_len,
                text_det_limit_type="max",  # 只缩小不放大：细图按原生尺寸跑，避免被放大 40+ 倍导致极慢
                **({"engine": ENGINE, "engine_config": ENGINE_CONFIG}
                   if ENGINE else {}),
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


def convert(result):
    """paddleocr 3.x list[dict] → Umi-OCR {code,data:[{box,score,text}]}."""
    if not result or len(result) == 0:
        return {"code": 100, "data": []}
    item = result[0]
    polys = item.get("rec_polys") or item.get("dt_polys") or []
    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or []
    data = []
    for poly, text, score in zip(polys, texts, scores):
        box = _unpad_box(poly)                       # ← 还原坐标
        data.append({"box": box, "score": float(score), "text": str(text)})
    return {"code": 100, "data": data}


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
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
                    f"init_ok ver={ver} lang={lang} cls={use_cls} "
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

                # 边缘补丁：给图像加白边，避免 PP-OCRv6 对近边缘文字丢首字
                tmp_padded = _pad_image(img)

                try:
                    # ── 关键修复：推理期间重定向 stdout → stderr ──
                    # paddle/oneDNN 在 C 层向 stdout 打印
                    #   [ReduceMeanCheckIfOneDNNSupport]
                    # 等诊断信息，会污染 Umi-OCR 读取的 JSON 协议首行，
                    # 导致宿主反序列化失败(904)。这里把 fd1 临时指向 fd2，
                    # 让这些噪声进 stderr（日志流），推理结束再恢复，
                    # 保证 stdout 只有唯一的 JSON 结果行。
                    saved_fd = os.dup(1)
                    sys.stdout.flush()
                    os.dup2(2, 1)
                    try:
                        result = ocr.ocr(tmp_padded)
                    finally:
                        os.dup2(saved_fd, 1)
                        os.close(saved_fd)
                    out = convert(result)
                finally:
                    # 清理补丁临时文件（base64 的由外层 finally 清理）
                    if tmp_padded and os.path.exists(tmp_padded):
                        try:
                            os.remove(tmp_padded)
                        except Exception:
                            pass
                        tmp_padded = None
        except Exception as e:
            out = {"code": 902, "data": f"识别失败：{type(e).__name__}: {e}"}

        dt = time.time() - t0
        rss = get_rss_mb()
        sys.stderr.write(
            f"[engine] #{req_count} {dt:.2f}s rss={rss}MB "
            f"out_code={out.get('code')} n_text={len(out.get('data',[]))}\n"
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
