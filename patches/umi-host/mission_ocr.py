# ===============================================
# =============== OCR - 任务管理器 ===============
# ===============================================

"""
一种任务管理器为全局单例，不同标签页要执行同一种任务，要访问对应的任务管理器。
任务管理器中有一个引擎API实例，所有任务均使用该API。
标签页可以向任务管理器提交一组任务队列，其中包含了每一项任务的信息，及总体的参数和回调。
"""

import os

from umi_log import logger
from .mission import Mission
from ..ocr.tbpu import getParser, IgnoreArea
from ..ocr.api import getApiOcr, getLocalOptions
from ..ocr.output.tools import ensure_geometry_table_on_res, promote_table_on_res
from ..ocr.tbpu.parser_tools.table_grid import build_table
from ..utils.utils import argdIntConvert

# 合法文件后缀
ImageSuf = [
    ".jpg",
    ".jpe",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
]


class __MissionOcrClass(Mission):
    def __init__(self):
        super().__init__()
        self._apiKey = ""  # 当前api类型
        self._api = None  # 当前引擎api对象

    # ========================= 【重载】 =========================

    # msnInfo: { 回调函数"onXX", 参数"argd":{"tbpu.xx", "ocr.xx"} }
    # msnList: [ { "path", "bytes", "base64" } ]
    def addMissionList(self, msnInfo, msnList):  # 添加任务列表
        # 实例化 tbpu 文本后处理模块
        msnInfo["tbpu"] = []
        argd = msnInfo["argd"]
        # 忽略区域
        if "tbpu.ignoreArea" in argd:
            iArea = argd["tbpu.ignoreArea"]
            if isinstance(iArea, list) and len(iArea) > 0:
                msnInfo["tbpu"].append(IgnoreArea(iArea))
        # 获取排版解析器对象
        if "tbpu.parser" in argd:
            msnInfo["tbpu"].append(getParser(argd["tbpu.parser"]))
        # 检查任务合法性
        for i in range(len(msnList) - 1, -1, -1):
            if "path" in msnList[i]:
                p = msnList[i]["path"]
                if os.path.splitext(p)[-1].lower() not in ImageSuf:
                    logger.warning(f"添加OCR任务时，第{i}项的路径path不是图片：{p}")
                    del msnList[i]
            elif "bytes" not in msnList[i] and "base64" not in msnList[i]:
                logger.warning(f"添加OCR任务时，第{i}项不含 path、bytes、base64")
                del msnList[i]
        if len(msnList) < 1:
            return "[Error] no valid mission in msnList!"
        return super().addMissionList(msnInfo, msnList)

    def msnPreTask(self, msnInfo):  # 用于更新api和参数
        # 检查API对象
        if not self._api:
            return "[Error] MissionOCR: API object is None."
        # 检查参数更新
        startInfo = self._dictShortKey(msnInfo["argd"])
        # 恢复int类型
        argdIntConvert(startInfo)
        msg = self._api.start(startInfo)
        if msg.startswith("[Error]"):
            logger.error(f"OCR引擎启动失败： {msg}")
            return msg  # 更新失败，结束该队列
        else:
            return ""  # 更新成功 TODO: continue

    def msnTask(self, msnInfo, msn):  # 执行msn
        if not self._api:
            return {
                "code": 901,
                "data": "[Error] MissionOCR: API object is None (engine stopped).",
            }
        try:
            if "path" in msn:
                res = self._api.runPath(msn["path"])
                res["path"] = msn["path"]  # 结果字典中补充参数
            elif "bytes" in msn:
                res = self._api.runBytes(msn["bytes"])
            elif "base64" in msn:
                res = self._api.runBase64(msn["base64"])
            else:
                res = {
                    "code": 901,
                    "data": f"[Error] Unknown task type.\n【异常】未知的任务类型。\n{str(msn)[:100]}",
                }
        except Exception as e:
            # 引擎被 stop/kill 时 run* 可能抛错；不得冲出工人线程
            logger.error("OCR msnTask 引擎调用异常。", exc_info=True, stack_info=True)
            return {
                "code": 902,
                "data": f"[Error] OCR engine call failed: {type(e).__name__}: {e}",
            }
        # 任务成功时的后处理
        if res.get("code") == 100:
            # 计算平均置信度
            score, num = 0, 0
            for r in res["data"]:
                score += r["score"]
                num += 1
            if num > 0:
                score /= num
            res["score"] = score
            # tableCsv 是结构请求：必须在普通排版 tbpu 改写原始 blocks 前建表。
            # 结构模型已返回 table 时 helper 会原样保留。
            res = ensure_geometry_table_on_res(
                res,
                bool(msnInfo["argd"].get("mission.filesType.tableCsv")),
                build_table,
            )
            # 执行 tbpu（与 DOC 侧一致：单页失败不得拖死工人）
            if msnInfo.get("tbpu"):
                for tbpu in msnInfo["tbpu"]:
                    try:
                        res["data"] = tbpu.run(res["data"])
                    except Exception as e:
                        logger.error(
                            f"OCR tbpu 异常 ({getattr(tbpu, 'tbpuName', type(tbpu).__name__)})",
                            exc_info=True,
                            stack_info=True,
                        )
                        res["code"] = 102
                        res["data"] = f"[Error] tbpu: {type(e).__name__}: {e}"
                        break
                    # 如果忽略区域等处理将所有文本删除，则结束tbpu
                    if not res["data"]:
                        res["code"] = 101
                        res["data"] = ""
                        break
            # 表格网格等：提升 last_table / _table → res["table"]
            res = promote_table_on_res(res, msnInfo.get("tbpu"))
        return res

    def _onStopRequested(self, msnIDs):
        # 停止时立刻杀掉引擎子进程，解除 runBytes 内的阻塞读
        self.interruptEngine("stop requested")

    def interruptEngine(self, reason=""):
        """强制结束 OCR 引擎进程（不抛错）。下次 msnPreTask 会重新 start。"""
        if not self._api:
            return
        logger.warning(f"MissionOCR.interruptEngine: {reason or 'interrupt'}")
        try:
            self._api.stop()
        except Exception:
            logger.error("MissionOCR.interruptEngine 失败。", exc_info=True)

    def forceRecover(self, reason=""):
        # 先杀引擎再清队列/工人，避免卡在 run*
        self.interruptEngine(reason or "forceRecover")
        super().forceRecover(reason or "forceRecover")

    # ========================= 【qml接口】 =========================

    def getStatus(self):  # 返回当前状态
        return {
            "apiKey": self._apiKey,
            "missionListsLength": self.getMissionListsLength(),
        }

    def setApi(self, apiKey, info):  # 设置api
        # 成功返回 [Success] ，失败返回 [Error] 开头的字符串
        self._apiKey = apiKey
        info = self._dictShortKey(info)
        # 如果api对象已启动，则先停止
        if self._api:
            self._api.stop()
        # 获取新api对象
        res = getApiOcr(apiKey, info)
        # 失败
        if isinstance(res, str):
            self._apiKey = ""
            self._api = None
            return res
        # 成功
        else:
            self._api = res
            return "[Success]"

    # 将字典中配置项的长key转为短key
    # 如： ocr.win32_PaddleOCR-json.path → path
    def _dictShortKey(self, d):
        newD = {}
        key1 = "ocr."
        key2 = key1 + self._apiKey + "."
        for k in d:
            if k.startswith(key2):
                newD[k[len(key2) :]] = d[k]
            elif k.startswith(key1):
                newD[k[len(key1) :]] = d[k]
        return newD

    # ========================= 【qml接口】 =========================

    def getLocalOptions(self):
        if self._apiKey:
            return getLocalOptions(self._apiKey)
        else:
            return {}


# 全局 OCR任务管理器
MissionOCR = __MissionOcrClass()
