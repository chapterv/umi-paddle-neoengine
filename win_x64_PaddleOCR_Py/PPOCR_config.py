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

# 三态选项：开启 / 关闭 / 使用全局参数（截图等窗口级覆盖全局）
_THREE_STATE = [
    ["on", "开启"],
    ["off", "关闭"],
    ["global", "使用全局参数"],
]


globalOptions = {
    # Part 2：统一后的全局标题（原 CPU 版“（新引擎）”/ GPU 版“（GPU·CUDA）”合并为一个）
    "title": tr("PaddleOCR（本地·PP-OCRv6/v5/v4）"),
    "type": "group",
    # ── 推理引擎（二级目录）：paddle / onnx-cpu / onnx-cuda，切换后下方相关项含义随之变化 ──
    "engine": {
        "title": tr("推理引擎"),
        "optionsList": [
            ["paddle", "Paddle (MKLDNN) CPU（默认）"],
            ["onnxruntime", "ONNX Runtime CPU"],
            ["onnxruntime-gpu", "ONNX Runtime CUDA GPU"],
        ],
        "default": "paddle",
        "toolTip": tr(
            "Paddle (MKLDNN)：Paddle 原生 oneDNN CPU 后端，3.2.1 下稳定，默认。\n"
            "ONNX Runtime CPU：绕过 oneDNN，可作对照 / 兜底。\n"
            "ONNX Runtime CUDA GPU：优先用 CUDA 提速（依赖随包装进 venv，"
            "无需系统装 CUDA Toolkit；不可用时自动回退 CPU）。"
        ),
    },
    # ── 选择模型版本：扁平项，紧挨推理引擎下方、版本回退链 ① 之前（R2-b，不打 group）──
    "ocr_version": {
        "title": tr("选择模型版本（最优先）"),
        "optionsList": [
            ["PP-OCRv6", "v6 medium（高精度·默认）"],
            ["PP-OCRv5", "v5（多语言·韩/俄）"],
            ["PP-OCRv4", "v4 mobile（快速）"],
        ],
        "default": "PP-OCRv6",
        "toolTip": tr(
            "用户选择的模型版本将作为**第一优先**尝试目标。\n"
            "若所选版本初始化失败，自动按下方「版本回退链」依次尝试其他版本。\n"
            "v6 medium：识别精度最高，简/繁/英/日文首选。\n"
            "v5：支持韩语/俄语等 PP-OCRv6 未覆盖的语言。\n"
            "v4 mobile：速度快约 2.7 倍，适合日常截图。"
        ),
    },
    # ── 版本回退链（用户选择版本失败后按此顺序尝试）──
    "fallback_1": {
        "title": tr("版本回退链 ①"),
        "default": "PP-OCRv6",
    },
    "fallback_2": {
        "title": tr("版本回退链 ②"),
        "default": "PP-OCRv5",
    },
    "fallback_3": {
        "title": tr("版本回退链 ③"),
        "default": "PP-OCRv4",
    },
    # ── MKL-DNN：仅 Paddle(MKLDNN) 后端生效；ONNX 自动忽略 ──
    "enable_mkldnn": {
        "title": tr("启用 MKL-DNN 加速"),
        "optionsList": [
            ["on", "开启"],
            ["off", "关闭"],
            ["na", "不适用·仅Paddle有效"],
        ],
        "default": "on",
        "toolTip": tr(
            "Intel oneDNN 加速，实测提速 2~3.5 倍（仅 Paddle(MKLDNN) 后端生效）。\n"
            "选 ONNX Runtime（CPU/GPU）时此选项不适用，引擎自动忽略。"
        ),
    },
    "cpu_threads": {
        "title": tr("线程数"),
        "default": _threads,
        "min": 1,
        "isInt": True,
    },
    # 内存占用 / 闲时清理：保持靠后（在线程数之后、文档预处理之前）
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
    # ── 文档预处理：三个功能打组放一起（全局基线）──
    "doc_preprocess": {
        "title": tr("文档预处理（方向纠正 / 去扭曲）"),
        "type": "group",
        "use_doc_orientation": {
            "title": tr("文档方向纠正"),
            "default": True,
            "toolTip": tr(
                "整图方向分类（0°/90°/180°/270°），自动纠正常见横屏/倒置的文档图片。"
                "默认开；引擎已做精确坐标逆映射，开启后检测框仍与原图完美对齐。"
            ),
        },
        "use_doc_unwarping": {
            "title": tr("文档去扭曲(矫正)"),
            "default": True,
            "toolTip": tr(
                "曲面文档展平（文档矫正），纠正弯曲/拍照畸变的书页。"
                "默认开；引擎已做精确坐标逆映射，开启后检测框仍与原图完美对齐。"
            ),
        },
        "use_textline_orientation": {
            "title": tr("纠正文本方向"),
            "default": True,
            "toolTip": tr(
                "逐行方向分类（0°/180°），纠正倒置的文本行（如竖排/反向扫描的中日韩文）。"
                "默认开；该开关只影响逐行文字朝向、不改变检测框坐标。"
            ),
        },
    },
}

localOptions = {
    "title": tr("文字识别（PaddleOCR·v6/v5/v4）"),
    "type": "group",
    "language": {
        "title": tr("语言/模型库"),
        "optionsList": _LanguageList,
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
    # ── 文档预处理（窗口级覆盖，默认『使用全局参数』）──
    "doc_preprocess_local": {
        "title": tr("文档预处理（覆盖全局）"),
        "type": "group",
        "use_doc_orientation": {
            "title": tr("文档方向纠正"),
            "optionsList": _THREE_STATE,
            "default": "global",
            "toolTip": tr("开启 / 关闭 / 使用全局参数。默认『使用全局参数』沿用全局设置。"),
        },
        "use_doc_unwarping": {
            "title": tr("文档去扭曲(矫正)"),
            "optionsList": _THREE_STATE,
            "default": "global",
            "toolTip": tr("开启 / 关闭 / 使用全局参数。默认『使用全局参数』沿用全局设置。"),
        },
        "use_textline_orientation": {
            "title": tr("纠正文本方向"),
            "optionsList": _THREE_STATE,
            "default": "global",
            "toolTip": tr("开启 / 关闭 / 使用全局参数。默认『使用全局参数』沿用全局设置。"),
        },
    },
}
