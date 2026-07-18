import os
import psutil
from plugin_i18n import Translator

tr = Translator(__file__, "i18n.csv")

# 模块配置路径
MODELS_CONFIGS = "/models/configs.txt"


# 动态获取模型库列表
def _getlanguageList():
    """configs.txt 格式示例：
    config_chinese.txt 简体中文
    config_en.txt English
    """
    optionsList = []
    configsPath = os.path.dirname(os.path.abspath(__file__)) + MODELS_CONFIGS
    try:
        with open(configsPath, "r", encoding="utf-8") as file:
            content = file.read()
            lines = content.split("\n")
            for l in lines:
                if not l.strip():  # 跳过空行（尾随换行会产生空元素，避免 parts[1] 越界）
                    continue
                parts = l.split(" ", 1)
                optionsList.append([f"models/{parts[0]}", parts[1]])
        return optionsList
    except FileNotFoundError:
        print(
            "[Error] PPOCR配置文件configs不存在，请检查文件路径是否正确。", configsPath
        )
    except IOError:
        print("[Error] PPOCR配置文件configs无法打开或读取。")
    return []


_LanguageList = _getlanguageList()


# 获取最佳线程数。用户设定可以覆盖这个计算值。
def _getThreads():
    threadsCount = 1
    try:
        phyCore = psutil.cpu_count(logical=False)  # 物理核心数
        lgiCore = psutil.cpu_count(logical=True)  # 逻辑核心数
        if (
            not isinstance(phyCore, int)
            or not isinstance(lgiCore, int)
            or lgiCore < phyCore
        ):
            raise ValueError("核心数计算异常")
        # 物理核数=逻辑核数，返回逻辑核数
        if phyCore * 2 == lgiCore or phyCore == lgiCore:
            threadsCount = lgiCore
        # 大小核处理器，返回大核线程数
        else:
            big = lgiCore - phyCore
            threadsCount = big * 2
        threadsCount = int(threadsCount)
    except Exception as e:
        print("[Warning] 无法获取CPU核心数！", e)
    # 线程上限16
    if threadsCount > 16:
        threadsCount = 16
    return threadsCount


_threads = _getThreads()


# 获取内存占用默认上限。用户设定可以覆盖这个计算值。
def _getRamMax():
    ramMax = 1024
    try:
        # 获取系统总内存数（以字节为单位）
        totalMemoryBytes = psutil.virtual_memory().total
        ramMax *= 0.5  # 取总内存的一半
        # 将总内存数转换为 MB 单位
        ramMax = totalMemoryBytes / 1048576
        ramMax = int(ramMax)
    except Exception as e:
        print("[Warning] 无法获取系统总内存数！", e)
    # 默认内存下限512MB，上限8G
    if ramMax < 512:
        ramMax = 512
    elif ramMax > 8192:
        ramMax = 8192
    return ramMax


_ramMax = _getRamMax()

globalOptions = {
    "title": tr("PaddleOCR·PP-OCRv6/v4（新引擎）"),
    "type": "group",
    "ocr_version": {
        "title": tr("模型版本"),
        "optionsList": [
            ["PP-OCRv4", "v4 mobile（快速·默认）"],
            ["PP-OCRv6", "v6 medium（高精度）"],
        ],
        "default": "PP-OCRv4",
        "toolTip": tr(
            "v4 mobile：速度快约 2.7 倍，适合日常截图/文档。\n"
            "v6 medium：识别精度更高，但速度较慢。"
        ),
    },
    "enable_mkldnn": {
        "title": tr("启用MKL-DNN加速"),
        "default": True,
        "toolTip": tr(
            "Intel oneDNN 加速，实测提速 2~3.5 倍（V6: 30s→8.5s, V4: 17s→9s）。"
            "需要 paddlepaddle==3.2.1（已锁定在 requirements.txt），"
            "3.3.x 版本存在 PIR 兼容性 bug 导致崩溃。"
        ),
    },
    "engine": {
        "title": tr("推理引擎"),
        "optionsList": [
            ["paddle", "Paddle (MKLDNN) 默认"],
            ["onnxruntime", "ONNX Runtime（绕过 oneDNN）"],
        ],
        "default": "paddle",
        "toolTip": tr(
            "Paddle (MKLDNN)：Paddle 原生后端，oneDNN 加速，3.2.1 下稳定。\n"
            "ONNX Runtime：绕过 oneDNN，可避免 3.3.x 的 MKLDNN 崩溃；"
            "速度相当，可作对照 / 兜底。"
        ),
    },
    "cpu_threads": {
        "title": tr("线程数"),
        "default": _threads,
        "min": 1,
        "isInt": True,
    },
    "ram_max": {
        "title": tr("内存占用限制"),
        "default": _ramMax,
        "min": -1,
        "unit": "MB",
        "isInt": True,
        "toolTip": tr("值>0时启用。引擎内存占用超过该值时，执行内存清理。"),
    },
    "ram_time": {
        "title": tr("内存闲时清理"),
        "default": 60,
        "min": -1,
        "unit": tr("秒"),
        "isInt": True,
        "toolTip": tr("值>0时启用。引擎空闲时间超过该值时，执行内存清理。"),
    },
}

localOptions = {
    "title": tr("文字识别（PaddleOCR·新·v4/v6）"),
    "type": "group",
    "language": {
        "title": tr("语言/模型库"),
        "optionsList": _LanguageList,
    },
    "cls": {
        "title": tr("纠正文本方向"),
        "default": False,
        "toolTip": tr("启用方向分类，识别倾斜或倒置的文本。可能降低识别速度。"),
    },
    "limit_side_len": {
        "title": tr("限制图像边长"),
        "optionsList": [
            [1920, "1920 " + tr("（默认·推荐）")],
            [960, "960 " + tr("（快速）")],
            [2880, "2880"],
            [4320, "4320"],
            [999999, tr("无限制")],
        ],
        "toolTip": tr(
            "将边长大于该值的图片进行压缩以提高速度。值越大精度越高但越慢；"
            "1920 适配大多数高 DPI 屏幕，960 适合纯速度优先场景。"
        ),
    },
}
