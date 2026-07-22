# ==============================================
# =============== 任务管理器 基类 ===============
# ==============================================


from PySide2.QtCore import QMutex, QRunnable
from threading import Condition
from uuid import uuid4  # 唯一ID
import time

from umi_log import logger
from ..utils.thread_pool import threadRun  # 异步执行函数


class Mission:
    def __init__(self):
        self._msnInfoDict = {}  # 任务信息的字典
        self._msnListDict = {}  # 任务队列的字典
        self._msnPausedDict = {}  # 已暂停的任务队列
        self._msnMutex = QMutex()  # 任务队列的锁
        self._task = None  # 异步任务对象
        self._taskMutex = QMutex()  # 任务对象的锁
        # 工人代数：forceRecover 时自增，旧工人退出时不得清掉新工人的 _task
        self._workerEpoch = 0
        # 任务队列调度方式
        # 1111 : 轮询调度，轮流取每个队列的第1个任务
        # 1234 : 顺序调度，将首个队列所有任务处理完，再进入下一个队列
        self._schedulingMode = "1111"

    # ========================= 【调用接口】 =========================

    """
    添加任务队列的格式
    mission = {
        "onStart": 任务队列开始回调函数 , (msnInfo)
        "onReady": 一项任务准备开始 , (msnInfo, msn)
        "onGet": 一项任务获取结果 , (msnInfo, msn, res)
        "onEnd": 任务队列结束 , (msnInfo, msg) // msg可选前缀： [Success] [Warning] [Error]
    }
    MissionOCR.addMissionList(mission, paths)
    """

    # 【异步】添加一条任务队列。成功返回任务ID，失败返回 startswith("[Error]")
    # msnInfo: { 回调函数 "onStart", "onReady", "onGet", "onEnd"}
    # msnList: [ 任务1, 任务2 ]
    def addMissionList(self, msnInfo, msnList):
        if len(msnList) < 1:
            return "[Error] no valid mission in msnList!"
        msnID = str(uuid4())
        # 检查并补充回调函数
        # 队列开始，单个任务准备开始，单任务取得结果，队列结束
        cbKeys = ["onStart", "onReady", "onGet", "onEnd"]
        for k in cbKeys:
            if k not in msnInfo or not callable(msnInfo[k]):
                msnInfo[k] = lambda *e: None
        # 任务状态state:  waiting 等待开始， running 进行中， stop 要求停止
        msnInfo["state"] = "waiting"
        msnInfo["msnID"] = msnID
        # 添加到任务队列
        self._msnMutex.lock()  # 上锁
        self._msnInfoDict[msnID] = msnInfo  # 添加任务信息
        self._msnListDict[msnID] = msnList  # 添加任务队列
        self._msnMutex.unlock()  # 解锁
        # 启动任务
        self._startMsns()
        # 返回任务id
        return msnID

    # 停止一些任务队列
    def stopMissionList(self, msnIDs):
        if not isinstance(msnIDs, list):
            msnIDs = [msnIDs]
        self._msnMutex.lock()  # 上锁
        for msnID in msnIDs:
            # 将暂停中的任务恢复
            if msnID in self._msnPausedDict:
                info, list_ = self._msnPausedDict[msnID]
                self._msnInfoDict[msnID] = info
                self._msnListDict[msnID] = list_
            # 将进行中的任务置为停止状态
            if msnID in self._msnListDict:
                self._msnInfoDict[msnID]["state"] = "stop"  # 设为停止状态
        self._msnMutex.unlock()  # 解锁
        self._startMsns()  # 拉起工作线程，使已暂停的任务可以正常结束
        # 子类可在 stop 后打断阻塞中的 msnTask（如杀 OCR 引擎）
        try:
            self._onStopRequested(msnIDs)
        except Exception:
            logger.error("Mission._onStopRequested 异常。", exc_info=True)

    # 停止全部任务
    def stopAllMissions(self):
        self._msnMutex.lock()  # 上锁
        # 将暂停中的任务恢复
        for msnID in self._msnPausedDict:
            info, list_ = self._msnPausedDict[msnID]
            self._msnInfoDict[msnID] = info
            self._msnListDict[msnID] = list_
        # 将进行中的任务置为停止状态
        for msnID in self._msnListDict:
            self._msnInfoDict[msnID]["state"] = "stop"
        self._msnMutex.unlock()  # 解锁
        self._startMsns()
        try:
            self._onStopRequested(None)
        except Exception:
            logger.error("Mission._onStopRequested 异常。", exc_info=True)

    # 暂停一些任务队列
    def pauseMissionList(self, msnIDs):
        if not isinstance(msnIDs, list):
            msnIDs = [msnIDs]
        self._msnMutex.lock()  # 上锁
        for msnID in msnIDs:
            if msnID in self._msnListDict:
                msn = (self._msnInfoDict[msnID], self._msnListDict[msnID])
                self._msnPausedDict[msnID] = msn
                del self._msnInfoDict[msnID]
                del self._msnListDict[msnID]
        self._msnMutex.unlock()  # 解锁
        logger.debug(f"任务暂停： {msnID}")

    # 恢复一些任务队列的运行
    def resumeMissionList(self, msnIDs):
        if not isinstance(msnIDs, list):
            msnIDs = [msnIDs]
        self._msnMutex.lock()  # 上锁
        for msnID in msnIDs:
            if msnID in self._msnPausedDict:
                info, list_ = self._msnPausedDict[msnID]
                self._msnInfoDict[msnID] = info
                self._msnListDict[msnID] = list_
                del self._msnPausedDict[msnID]
        self._msnMutex.unlock()  # 解锁
        self._startMsns()  # 拉起工作线程
        logger.debug(f"任务恢复： {msnID}")

    # 获取每一条任务队列长度
    def getMissionListsLength(self):
        lenDict = {}
        self._msnMutex.lock()
        for k in self._msnListDict:
            lenDict[str(k)] = len(self._msnListDict[k])
        self._msnMutex.unlock()
        return lenDict

    # 【同步】添加一个任务或队列，等待完成，返回任务结果列表。[i]["result"]为结果
    # timeout: 最长等待秒数；None 表示不限总时长（仍每秒醒来一次，便于被 stop 打断）
    def addMissionWait(self, argd, msnList, timeout=None):
        if not isinstance(msnList, list):
            msnList = [msnList]
        resList = msnList[:]  # 浅拷贝出一条结果列表
        nowIndex = 0  # 当前处理的任务
        msnLen = len(msnList)
        condition = Condition()  # 线程同步器
        endMsg = ""  # 任务结束的消息
        finished = False

        def _onGet(msnInfo, msn, res):
            nonlocal nowIndex
            resList[nowIndex]["result"] = res
            nowIndex += 1

        def _onEnd(msnInfo, msg):
            nonlocal endMsg, finished
            endMsg = msg
            finished = True
            with condition:  # 释放线程阻塞
                condition.notify()

        def _pass(*x):
            pass

        msnInfo = {
            "onStart": _pass,
            "onReady": _pass,
            "onGet": _onGet,
            "onEnd": _onEnd,
            "argd": argd,
        }
        msnID = self.addMissionList(msnInfo, msnList)
        if msnID.startswith("[Error]"):  # 添加任务失败
            endMsg = msnID
            finished = True
        else:
            deadline = (time.time() + timeout) if timeout is not None else None
            # 分段 wait：可被 stop/timeout 打断，避免永久卡死父任务（如 MissionDOC）
            while not finished:
                if deadline is not None and time.time() >= deadline:
                    logger.error(
                        f"addMissionWait 超时（>{timeout}s），强制 stopMissionList。msnID={msnID}"
                    )
                    try:
                        self.stopMissionList(msnID)
                    except Exception:
                        logger.error("addMissionWait 超时 stop 失败。", exc_info=True)
                    # 再给一点时间让 onEnd 到来；仍未结束则强制收尾
                    with condition:
                        condition.wait(timeout=3.0)
                    if not finished:
                        endMsg = f"[Error] Mission wait timeout (>{timeout}s)."
                        finished = True
                        # 尽量从队列摘掉，避免僵尸任务
                        self._msnMutex.lock()
                        if msnID in self._msnInfoDict:
                            try:
                                self._msnInfoDict[msnID]["onEnd"] = _pass
                            except Exception:
                                pass
                            self._msnDictDel(msnID)
                        self._msnMutex.unlock()
                    break
                with condition:
                    condition.wait(timeout=1.0)
        # 补充未完成的任务
        for i in range(nowIndex, msnLen):
            if "result" not in resList[i]:
                resList[i]["result"] = {"code": 803, "data": f"任务提前结束。{endMsg}"}
        return resList

    # ========================= 【卡死恢复】 =========================

    def forceRecover(self, reason=""):
        """强制清空队列并释放工人槽位，使停止后可再次提交任务。

        场景：msnTask 阻塞（如 OCR 引擎挂死、addMissionWait 永不返回）时，
        stop 只设 state=stop 但工人进不了检查点 → `_task` 永不为 None → 新任务假死。

        注意：不调用 stopAllMissions（避免与子类 stop 钩子互相递归）；
        直接标 stop、清队列、打断底层、作废工人。
        """
        reason = reason or "forceRecover"
        logger.warning(f"Mission.forceRecover: {reason} ({type(self).__name__})")
        # 1) 全部标 stop（不经 stopAll，避免递归）
        self._msnMutex.lock()
        try:
            for msnID in self._msnPausedDict:
                info, list_ = self._msnPausedDict[msnID]
                self._msnInfoDict[msnID] = info
                self._msnListDict[msnID] = list_
            self._msnPausedDict.clear()
            for msnID in self._msnInfoDict:
                self._msnInfoDict[msnID]["state"] = "stop"
        finally:
            self._msnMutex.unlock()
        # 2) 打断底层阻塞（杀引擎等）
        try:
            self._onStopRequested(None)
        except Exception:
            logger.error("forceRecover._onStopRequested 异常。", exc_info=True)
        # 3) 清空所有队列，并回调 onEnd（避免 UI 永远等）
        pending = []
        self._msnMutex.lock()
        try:
            for msnID in list(self._msnInfoDict.keys()):
                pending.append(self._msnInfoDict[msnID])
            self._msnInfoDict.clear()
            self._msnListDict.clear()
            self._msnPausedDict.clear()
        finally:
            self._msnMutex.unlock()
        for info in pending:
            try:
                info["state"] = "stop"
                info["onEnd"](info, f"[Warning] Task force recovered: {reason}")
            except Exception:
                logger.error("forceRecover onEnd 回调异常。", exc_info=True)
        # 4) 作废旧工人，允许 _startMsns 再拉新工人
        self._forceResetWorker()

    def _forceResetWorker(self):
        self._taskMutex.lock()
        self._workerEpoch += 1
        self._task = None
        self._taskMutex.unlock()

    def isWorkerBusy(self):
        self._taskMutex.lock()
        busy = self._task is not None
        self._taskMutex.unlock()
        return busy

    # ========================= 【主线程 方法】 =========================

    def _startMsns(self):  # 启动异步任务，执行所有任务列表
        # 若当前异步任务对象为空，则创建工作线程
        self._taskMutex.lock()  # 上锁
        if self._task == None:
            self._workerEpoch += 1
            epoch = self._workerEpoch
            self._task = threadRun(self._taskRun, epoch)
        self._taskMutex.unlock()  # 解锁

    # ========================= 【子线程 方法】 =========================

    def _taskRun(self, epoch=None):  # 异步执行任务字典的流程
        # docs/11：msnTask 未捕获异常会冲出本函数，导致 _taskFinish 永不执行，
        # MissionDOC 工人假死（批量/预览 OCR 起不来，截图 OCR 仍正常）。
        if epoch is None:
            epoch = self._workerEpoch
        try:
            dictIndex = 0  # 当前取任务字典中的第几个任务队列
            # 循环，直到任务队列的列表为空
            while True:
                # 0. 旧工人被 forceRecover 作废 → 立刻退出
                if epoch != self._workerEpoch:
                    logger.warning(
                        f"Mission 旧工人退出（epoch {epoch}!={self._workerEpoch}）"
                    )
                    break

                # 1. 检查api和任务字典是否为空
                self._msnMutex.lock()  # 锁1 上锁
                dl = len(self._msnInfoDict)  # 任务字典长度
                if dl == 0:  # 任务字典已空
                    self._msnMutex.unlock()  # 锁1 解锁
                    break

                # 2. 任务调度，取一个任务
                if self._schedulingMode == "1111":  # 轮询
                    dictIndex = (dictIndex + 1) % dl
                elif self._schedulingMode == "1234":  # 顺序
                    dictIndex = 0  # 始终为首个队列
                dictKey = tuple(self._msnInfoDict.keys())[dictIndex]
                msnInfo = self._msnInfoDict[dictKey]
                msnList = self._msnListDict[dictKey]
                self._msnMutex.unlock()  # 锁1 解锁

                # 3. 检查任务是否要求停止
                if msnInfo["state"] == "stop":
                    self._msnDictDel(dictKey)
                    msnInfo["onEnd"](msnInfo, "[Warning] Task stop.")
                    continue

                # 4. 前处理，检查、更新参数
                preFlag = self.msnPreTask(msnInfo)
                if preFlag == "continue":  # 跳过本次
                    logger.debug(f"任务跳过： {dictKey}")
                    continue
                elif preFlag.startswith("[Error]"):  # 异常，结束该队列
                    msnInfo["onEnd"](msnInfo, preFlag)
                    self._msnDictDel(dictKey)
                    dictIndex -= 1  # 字典下标回退1位，下次执行正确的下一项
                    continue

                # 5. 首次任务
                if msnInfo["state"] == "waiting":
                    msnInfo["state"] = "running"
                    msnInfo["onStart"](msnInfo)

                # 6. 执行任务，并记录时间
                msn = msnList[0]
                msnInfo["onReady"](msnInfo, msn)
                t1 = time.time()
                try:
                    res = self.msnTask(msnInfo, msn)
                except Exception as e:
                    # 单页异常：不拖死整条队列与工人线程（如空白页 tbpu median 崩溃）
                    logger.error(
                        f"msnTask 异常，本页记错误并继续。key={dictKey} msn={msn!r}",
                        exc_info=True,
                        stack_info=True,
                    )
                    res = {
                        "code": 102,
                        "data": f"[Error] msnTask exception: {type(e).__name__}: {e}",
                    }
                t2 = time.time()
                if isinstance(res, dict):  # 补充耗时和时间戳
                    res["time"] = t2 - t1
                    res["timestamp"] = t2

                # 旧工人在长时间 msnTask 后被作废：丢弃结果并退出
                if epoch != self._workerEpoch:
                    logger.warning(
                        f"Mission 旧工人在 msnTask 后退出（epoch {epoch}!={self._workerEpoch}）"
                    )
                    break

                # 7. 再次检查任务是否要求停止，或者已暂停
                self._msnMutex.lock()  # 锁2 上锁
                if msnInfo["state"] == "stop":
                    self._msnDictDel(dictKey)
                    self._msnMutex.unlock()  # 锁2 解锁
                    msnInfo["onEnd"](msnInfo, "[Warning] Task stop.")
                    continue
                if dictKey not in self._msnInfoDict:
                    self._msnMutex.unlock()  # 锁2 解锁
                    continue

                # 8. 不停止，则上报该任务
                msnList.pop(0)  # 弹出该任务
                self._msnMutex.unlock()  # 锁2 解锁
                # 回调。注意：回调函数执行时间长时，可能用户再次提交了任务暂停，需要后续继续判断。
                msnInfo["onGet"](msnInfo, msn, res)

                # 9. 这条任务队列完成
                if len(msnList) == 0:
                    msnInfo["onEnd"](msnInfo, "[Success]")
                    self._msnMutex.lock()  # 锁3 上锁
                    self._msnDictDel(dictKey)
                    self._msnMutex.unlock()  # 锁3 解锁
                    dictIndex -= 1  # 字典下标回退1位，下次执行正确的下一项

        except Exception:
            # 调度循环自身异常：仍须复位 _task，否则 MissionDOC/预览 OCR 永久起不来
            logger.error(
                "Mission._taskRun 调度循环异常，强制结束工人。",
                exc_info=True,
                stack_info=True,
            )
        finally:
            # 完成 / 异常：必须清 _task，允许后续 _startMsns 再拉起工人
            self._taskFinish(epoch)

    def _msnDictDel(self, dictKey):  # 停止一组任务队列
        # 正常 删除任务队列项
        if dictKey in self._msnInfoDict:
            del self._msnInfoDict[dictKey]
            del self._msnListDict[dictKey]
        # 如果该任务在暂停中，则移除暂停队列中的项
        if dictKey in self._msnPausedDict:
            del self._msnPausedDict[dictKey]
            logger.debug(f"移除暂停任务： {dictKey}")

    def _taskFinish(self, epoch=None):  # 任务结束
        self._taskMutex.lock()  # 上锁
        # 仅本代工人可清空槽位，避免旧工人 finally 抹掉新工人
        if epoch is None or epoch == self._workerEpoch:
            self._task = None
        self._taskMutex.unlock()  # 解锁

    # ========================= 【继承重载】 =========================

    def _onStopRequested(self, msnIDs):
        """stop 后钩子：打断可能阻塞的底层资源。默认无操作。"""
        return

    def msnPreTask(self, msnInfo):  # 任务前处理，用于更新api和参数。
        """返回值可选：
        "" ：空字符串表示正常继续。
        "continue" ：跳过本次任务
        "[Error] xxxx" ：终止这条任务队列，返回异常信息
        """
        # return "[Error] No overloaded msnPreTask. \n【异常】未重载msnPreTask。"
        return ""

    def msnTask(self, msnInfo, msn):  # 执行任务msn，返回结果字典。
        logger.debug("mission 未重载 msnTask")
        return {"error": "[Error] No overloaded msnTask. \n【异常】未重载 msnTask。"}

    def getStatus(self):  # 返回当前状态
        return "Mission 基类 返回空状态"
