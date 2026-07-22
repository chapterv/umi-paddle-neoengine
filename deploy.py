# -*- coding: utf-8 -*-
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(ROOT, "Umi-OCR", "UmiOCR-data", "plugins", "win_x64_PaddleOCR_Py")
PY = os.path.join(PLUGIN, ".venv_gpu", "Scripts", "python.exe")
CACHE = os.path.join(PLUGIN, "paddlex")
os.environ["PADDLE_PDX_CACHE_HOME"] = CACHE

SELF_TEST = "--selftest" in sys.argv

MENU = (
    "\n[3/3] 选择预下载的模型范围：\n"
    "[1] 完整版   : V4 + V6 + V5多语言 · ONNX + MKL-DNN  ~最大\n"
    "[2] 最小可用 : PP-OCRv6（中文）· 仅 ONNX            ~最小\n"
    "[3] 多语言   : V6 + V5多语言 · 仅 ONNX              ~中等\n"
    "（直接回车 = 选项2）\n"
)

# 每个选项：
#   engines = 要下的引擎版本列表（"paddle"=MKLDNN / "onnx"=ONNX Runtime）
#   ocr     = 要下的 (ocr_version, lang) 列表
# 三个工具模型（文档方向 / 文本行方向 / 文档去扭曲）在每个选项里都下载：
#   随该引擎的【最后一个 ocr 对】一起实例化（with_tools=True），保证必下。
SCOPES = {
    "1": {
        "engines": ["paddle", "onnx"],
        "ocr": [("PP-OCRv4", "ch"), ("PP-OCRv6", "ch"),
                ("PP-OCRv5", "eslav"), ("PP-OCRv5", "korean")],
    },
    "2": {
        "engines": ["onnx"],
        "ocr": [("PP-OCRv6", "ch")],
    },
    "3": {
        "engines": ["onnx"],
        "ocr": [("PP-OCRv6", "ch"), ("PP-OCRv5", "eslav"), ("PP-OCRv5", "korean")],
    },
}


def log(msg):
    print(msg, flush=True)


def download(ver, lang, engine, with_tools):
    """下载某 (ocr_version, lang) 的模型。
    engine='onnx' 下 ONNX 版；engine='paddle' 下 PADDLE(MKLDNN) 版。
    with_tools=True 时一并强制下载三个工具模型
    （use_doc_orientation_classify / use_doc_unwarping / use_textline_orientation 全开）。
    注：PP-OCRv5 默认检测模型即 PP-OCRv5_server_det，
    故 V5 调用会顺带拉下 server-det（选项3 所需）。
    """
    tag = "onnx" if engine == "onnx" else "paddle"
    log("[下载] %s / lang=%s (engine=%s%s) ..." % (
        ver, lang, tag, " + 工具模型" if with_tools else ""))
    if SELF_TEST:
        log("  [OK-selftest] %s/%s/%s 跳过真实下载" % (ver, lang, tag))
        return
    try:
        import paddleocr
        kwargs = dict(
            device="cpu", lang=lang, ocr_version=ver,
            # 工具模型默认关（textline）/开（doc_*）；下工具时统一强制全开，
            # 确保 doc方向 / textline方向 / 文档去扭曲 三个 onnx 都落地。
            use_textline_orientation=with_tools,
            use_doc_orientation_classify=with_tools,
            use_doc_unwarping=with_tools,
            enable_mkldnn=(engine == "paddle"),
            cpu_threads=6,
        )
        if engine == "onnx":
            kwargs["engine"] = "onnxruntime"
            kwargs["engine_config"] = {"providers": ["CPUExecutionProvider"]}
        paddleocr.PaddleOCR(**kwargs)
        log("  [OK] %s/%s/%s 已就绪。" % (ver, lang, tag))
    except Exception as e:
        log("  [提示] %s/%s/%s 未完成（可能离线），首次识别会自动下载：%s"
            % (ver, lang, tag, e))


def main():
    if SELF_TEST:
        log("[selftest] 路径检查:")
        log("  ROOT   = %s" % ROOT)
        log("  PLUGIN = %s" % PLUGIN)
        log("  PY     = %s" % PY)
        log("  CACHE  = %s" % CACHE)
        log("[selftest] 菜单渲染（应正确显示中文）:")
        log(MENU)
        for ch in ("1", "2", "3"):
            spec = SCOPES[ch]
            log("[selftest] scope %s: engines=%s ocr=%s" % (ch, spec["engines"], spec["ocr"]))
        log("[selftest] OK")
        return

    choice = None
    if "--choice" in sys.argv:
        i = sys.argv.index("--choice")
        if i + 1 < len(sys.argv):
            choice = sys.argv[i + 1].strip()
    if choice not in SCOPES:
        log(MENU)
        try:
            raw = input("请输入 1/2/3（默认 2）: ")
        except EOFError:
            raw = ""
        choice = raw.strip() or "2"
        if choice not in SCOPES:
            choice = "2"
    spec = SCOPES[choice]
    log("（按选项 %s 下载）" % choice)
    for engine in spec["engines"]:
        n = len(spec["ocr"])
        for idx, (ver, lang) in enumerate(spec["ocr"]):
            # 工具模型随该引擎最后一个 ocr 对一起下（保证三个工具模型必下）
            download(ver, lang, engine, with_tools=(idx == n - 1))


if __name__ == "__main__":
    main()
