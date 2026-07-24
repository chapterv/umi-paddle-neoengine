# ========================================
# =============== 批量OCR页 ===============
# ========================================

import os
import time

from umi_log import logger
from .page import Page
from ..mission.mission_ocr import MissionOCR
from ..utils.utils import allowedFileName
from ..ocr.output import Output
from ..ocr.output.tools import resolve_table_request_task, resolve_trace_capture_path


class BatchOCR(Page):
    def __init__(self, *args):
        super().__init__(*args)
        self.argd = None
        self.msnID = ""
        self.outputList = []

    def msnPaths(self, paths, argd):
        msnInfo = {
            "onStart": self._onStart,
            "onReady": self._onReady,
            "onGet": self._onGet,
            "onEnd": self._onEnd,
            "argd": argd,
            # P1 capability is used only by an explicit table.csv request.
            "request_task": resolve_table_request_task(argd),
        }
        if not self._preprocessArgd(argd, paths[0]):
            return ""
        if not self._initOutputList(argd):
            return ""
        msnList = [{"path": x} for x in paths]
        self.msnID = MissionOCR.addMissionList(msnInfo, msnList)
        if self.msnID.startswith("[Error]"):
            self._onEnd(None, f"{self.msnID}\n添加任务失败。")
        else:
            logger.debug(f"添加任务成功 {self.msnID}")
        return self.msnID

    def _preprocessArgd(self, argd, path0):
        self.argd = None
        if argd["mission.dirType"] == "source":
            argd["mission.dir"] = os.path.dirname(path0)
        else:
            d = os.path.abspath(argd["mission.dir"])
            if not os.path.exists(d):
                try:
                    os.makedirs(d)
                except OSError:
                    logger.warning(f"批量OCR无法创建目录： {d}", exc_info=True)
                    self._onEnd(None, f'[Error] Failed to create directory: "{d}"\n【异常】无法创建目录。')
                    return False
            argd["mission.dir"] = d
        startTimestamp = time.time()
        argd["startTimestamp"] = startTimestamp
        argd["startDatetime"] = time.strftime(
            r"%Y-%m-%d %H:%M:%S", time.localtime(startTimestamp)
        )
        startDatetimeUser = argd["mission.datetimeFormat"].replace(
            r"%unix", str(startTimestamp)
        )
        startDatetimeUser = time.strftime(
            startDatetimeUser, time.localtime(startTimestamp)
        )
        fileName = argd["mission.fileNameFormat"]
        fileName = fileName.replace(r"%date", startDatetimeUser)
        fileNameEle = os.path.basename(os.path.dirname(path0))
        fileName = fileName.replace("%name", fileNameEle)
        if not allowedFileName(fileName):
            self._onEnd(None, f'[Error] The file name is illegal.\n【错误】文件名【{fileName}】含有不允许的字符。\n不允许含有下列字符： \\  /  :  *  ?  "  <  >  |')
            return False
        argd["mission.fileName"] = fileName
        self.argd = argd
        return True

    def _initOutputList(self, argd):
        self.outputList = []
        outputArgd = {
            "outputDir": argd["mission.dir"],
            "outputDirType": argd["mission.dirType"],
            "outputFileName": argd["mission.fileName"],
            "startDatetime": argd["startDatetime"],
            "ignoreBlank": argd["mission.ignoreBlank"],
            "traceCapturePath": resolve_trace_capture_path(argd),
        }
        try:
            for key in argd.keys():
                if "mission.filesType" in key and argd[key]:
                    self.outputList.append(Output[key[18:]](outputArgd))
        except Exception as e:
            self._onEnd(None, f"[Error] Failed to initialize output file.\n【错误】初始化输出文件失败。\n{e}")
            return False
        return True

    def msnStop(self):
        MissionOCR.stopMissionList(self.msnID)

    def msnPause(self):
        MissionOCR.pauseMissionList(self.msnID)

    def msnResume(self):
        MissionOCR.resumeMissionList(self.msnID)

    def msnPreview(self, path, argd):
        msnInfo = {"onGet": self._onPreview, "argd": argd}
        msnList = [{"path": path}]
        self.msnID = MissionOCR.addMissionList(msnInfo, msnList)

    def _onStart(self, msnInfo):
        pass

    def _onReady(self, msnInfo, msn):
        msnID = msnInfo["msnID"]
        if msnID != self.msnID:
            logger.warning(f"_onReady 任务ID未在记录。{msnID}")
            return
        self.callQmlInMain("onOcrReady", msn["path"])

    def _onGet(self, msnInfo, msn, res):
        msnID = msnInfo["msnID"]
        if msnID != self.msnID:
            logger.warning(f"_onGet 任务ID未在记录。{msnID}")
            return
        res["fileName"] = os.path.basename(msn["path"])
        res["dir"] = os.path.dirname(msn["path"])
        for o in self.outputList:
            try:
                o.print(res)
            except Exception:
                logger.error(f"结果输出失败：{o}", exc_info=True, stack_info=True)
        self.callQmlInMain("onOcrGet", msn["path"], res)

    def _onEnd(self, msnInfo, msg):
        if msnInfo:
            msnID = msnInfo["msnID"]
            if msnID != self.msnID:
                logger.warning(f"_onEnd 任务ID未在记录。{msnID}")
                return
        else:
            msnID = ""
        for o in self.outputList:
            try:
                o.onEnd()
            except Exception as e:
                msg = f"[Error] 输出器异常：{e}" + msg
        self.callQmlInMain("onOcrEnd", msg, msnID)

    def _onPreview(self, msnInfo, msn, res):
        self.callQmlInMain("onPreview", msn["path"], res)
