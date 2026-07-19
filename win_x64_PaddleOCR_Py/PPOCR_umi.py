# Umi-OCR 插件接口： PaddleOCR（Python 引擎 / Route B）
# 引擎入口 run.cmd 会调用本插件内 .venv 的 python 运行 engine.py
# 协议与 hiroi-sora/PaddleOCR-json 完全兼容，Umi-OCR 主程序零改动。

import os
import psutil  # 进程检查
from platform import system  # 平台检查

from call_func import CallFunc
from .PPOCR_api import PPOCR_pipe

# 引擎可执行文件（入口）名称
# 本插件用 run.cmd 作为入口，由它调用 .venv 内的 python 运行 engine.py
system_type = system()
ExeFile = ""
if system_type == "Windows":
    ExeFile = "run.cmd"
elif system_type == "Linux":
    ExeFile = "run.sh"
else:
    raise NotImplementedError(f"[Error] PaddleOCR: Unsupported system: {system_type}")

# 引擎可执行文件路径
ExePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ExeFile)


class Api:  # 公开接口
    def __init__(self, globalArgd):
        # 测试路径是否存在
        if not os.path.exists(ExePath):
            raise ValueError(f'[Error] Exe path "{ExePath}" does not exist.')
        # 保存全局配置，供 start() 做「局部三态 → 全局」继承解析
        self.api = None  # api对象
        self._globalArgd = globalArgd
        self.exeConfigs = {}  # exe启动参数字典（基线，按全局解析）
        # 内存清理参数
        self.ramInfo = {"max": -1, "time": -1, "timerID": ""}
        m = globalArgd.get("ram_max")
        if isinstance(m, (int, float)):
            self.ramInfo["max"] = m
        m = globalArgd.get("ram_time")
        if isinstance(m, (int, float)):
            self.ramInfo["time"] = m
        self.isInit = True

    def _resolve(self, globalData, localData):
        """把全局 + 局部配置解析成 engine.py 需要的启动参数字典。

        - 全局基线：engine / ocr_version / cpu_threads / enable_mkldnn /
          fallback_1~3 / doc_preprocess.* （三组文档预处理开关，bool）
        - 局部覆盖（截图等窗口级）：language / limit_side_len /
          doc_preprocess_local.* （三态：on/off/global）
        - 三态 'global' 或缺失 → 沿用全局 bool；'on'→True；'off'→False。
        """
        cfg = {}
        # ── 直接透传（全局项）──
        if "engine" in globalData:
            cfg["engine"] = globalData["engine"]
        # ocr_version：扁平键优先；兼容曾用 group 时的 model_version.ocr_version
        _ver = globalData.get("ocr_version") or globalData.get(
            "model_version.ocr_version"
        )
        if _ver:
            cfg["ocr_version"] = _ver
        if "cpu_threads" in globalData:
            cfg["cpu_threads"] = globalData["cpu_threads"]
        # ── 语言 / 限制边长（局部项，仅窗口级有）──
        lang = (localData or {}).get("language")
        if lang:
            cfg["config_path"] = lang
        side = (localData or {}).get("limit_side_len")
        if side is not None:
            cfg["limit_side_len"] = side
        # ── MKL-DNN 三态（仅 Paddle(MKLDNN) 后端生效；ONNX 引擎自动忽略）──
        # on→True / off→False / na→不传（引擎默认 True 给 paddle，ONNX 强制 False）
        mkl = globalData.get("enable_mkldnn", "on")
        if mkl == "on":
            cfg["enable_mkldnn"] = True
        elif mkl == "off":
            cfg["enable_mkldnn"] = False
        # ── 版本回退链（扁平 fallback_*；兼容旧 group 键）──
        chain = []
        for i in (1, 2, 3):
            c = globalData.get(f"fallback_{i}") or globalData.get(
                f"version_fallback.fallback_{i}", ""
            )
            if c:
                chain.append(c)
        if chain:
            cfg["fallback_chain"] = ",".join(chain)
        # ── 文档预处理：全局基线 + 局部三态覆盖 ──
        g_ori = globalData.get("doc_preprocess.use_doc_orientation", True)
        g_unwarp = globalData.get("doc_preprocess.use_doc_unwarping", True)
        g_line = globalData.get("doc_preprocess.use_textline_orientation", True)

        def _resolve3(localKey, gb):
            lv = (localData or {}).get(localKey)
            if lv == "on":
                return True
            if lv == "off":
                return False
            # "global"（默认）或缺失 → 沿用全局基线
            return gb

        cfg["use_doc_orientation"] = _resolve3(
            "doc_preprocess_local.use_doc_orientation", g_ori
        )
        cfg["use_doc_unwarping"] = _resolve3(
            "doc_preprocess_local.use_doc_unwarping", g_unwarp
        )
        cfg["use_textline_orientation"] = _resolve3(
            "doc_preprocess_local.use_textline_orientation", g_line
        )
        return cfg

    # 启动引擎。返回： "" 成功，"[Error] xxx" 失败
    def start(self, argd):
        # 解析启动参数（全局基线 + 局部覆盖）
        tempConfigs = self._resolve(self._globalArgd, argd)
        # 若引擎已启动，且参数与传入一致，则无需重启
        if self.api is not None:
            if set(tempConfigs.items()) == set(self.exeConfigs.items()):
                return ""
            # 若引擎已启动但需要更改参数，则停止旧引擎
            self.stop()
        # 启动新引擎
        self.exeConfigs = tempConfigs
        try:
            self.api = PPOCR_pipe(ExePath, argument=tempConfigs)
        except Exception as e:
            self.api = None
            return f"[Error] OCR init fail. Argd: {tempConfigs}\n{e}"
        return ""

    def stop(self):  # 停止引擎（必须不抛错，否则语言切换/重配会卡死任务线程）
        if self.api is None:
            return
        try:
            self.api.exit()
        except Exception:
            pass
        self.api = None

    def runPath(self, imgPath: str):  # 路径识图
        self.__runBefore()
        res = self.api.run(imgPath)
        self.__ramClear()
        return res

    def runBytes(self, imageBytes):  # 字节流
        self.__runBefore()
        res = self.api.runBytes(imageBytes)
        self.__ramClear()
        return res

    def runBase64(self, imageBase64):  # base64字符串
        self.__runBefore()
        res = self.api.runBase64(imageBase64)
        self.__ramClear()
        return res

    def __runBefore(self):
        CallFunc.delayStop(self.ramInfo["timerID"])  # 停止ram清理计时器

    def _restart(self):  # 重启引擎
        self.stop()
        # 启动引擎
        try:
            self.api = PPOCR_pipe(ExePath, argument=self.exeConfigs)
            print("重启引擎")
        except Exception as e:
            self.api = None
            print(f"[Error]重启引擎失败: {e}")

    def __ramClear(self):  # 内存清理
        if self.ramInfo["max"] > 0:
            pid = self.api.ret.pid
            rss = psutil.Process(pid).memory_info().rss
            rss /= 1048576
            if rss > self.ramInfo["max"]:
                self._restart()
        if self.ramInfo["time"] > 0:
            self.ramInfo["timerID"] = CallFunc.delay(
                self._restart, self.ramInfo["time"]
            )
