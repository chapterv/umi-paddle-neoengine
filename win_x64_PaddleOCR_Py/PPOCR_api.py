# 调用 PaddleOCR（Python 引擎）的 Python Api
# 协议兼容 hiroi-sora/PaddleOCR-json 的管道模式：
#   启动后向 stdout 打印 "OCR init completed." 表示就绪；
#   每行 stdin 输入 {"image_path":...} 或 {"image_base64":...}；
#   每行 stdout 输出 {"code":100,"data":[{"box":...,"score":...,"text":...}]}
# 本文件运行于 Umi-OCR 自带的 Python 环境，不依赖 paddle。

import os
import atexit  # 退出处理
import subprocess  # 进程，管道
import time
import threading
import queue
from json import loads as jsonLoads, dumps as jsonDumps
from sys import platform as sysPlatform  # popen静默模式
from base64 import b64encode  # base64 编码

# 超时（秒）
_INIT_TIMEOUT = 180   # 引擎 init：覆盖多版本回退(每版本90s)+模型加载
_READ_TIMEOUT = 120   # 单次 OCR 推理读结果；超时须 kill 引擎并返回，避免 UI 永久卡死


class PPOCR_pipe:  # 调用OCR（管道模式）
    def __init__(self, exePath: str, modelsPath: str = None, argument: dict = None):
        """初始化识别器（管道模式）。\n
        `exePath`: 识别器入口（run.cmd）。\n
        `modelsPath`: 识别库`models`文件夹的路径。若为None则默认识别库与识别器在同一目录下。\n
        `argument`: 启动参数，字典`{"键":值}`。\n
        """
        # 私有成员变量
        self.__ENABLE_CLIPBOARD = False
        self.ret = None
        self._out_q = None          # 唯一 stdout 行队列（init + 推理共用）
        self._reader_thread = None
        self._stderr_fd = None

        exePath = os.path.abspath(exePath)
        cwd = os.path.abspath(os.path.join(exePath, os.pardir))  # 获取exe父文件夹
        cmds = [exePath]
        # 处理启动参数
        if modelsPath is not None:
            if os.path.exists(modelsPath) and os.path.isdir(modelsPath):
                cmds += ["--models_path", os.path.abspath(modelsPath)]
            else:
                raise Exception(
                    f"Input modelsPath doesn't exits or isn't a directory. modelsPath: [{modelsPath}]"
                )
        if isinstance(argument, dict):
            for key, value in argument.items():
                # Popen() 要求输入list里所有的元素都是 str 或 bytes
                if isinstance(value, bool):
                    cmds += [f"--{key}={value}"]  # 布尔参数必须键和值连在一起
                elif isinstance(value, str):
                    cmds += [f"--{key}", value]
                else:
                    cmds += [f"--{key}", str(value)]
        # 设置子进程启用静默模式，不显示控制台窗口
        startupinfo = None
        if "win32" in str(sysPlatform).lower():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags = (
                subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
            )
            startupinfo.wShowWindow = subprocess.SW_HIDE
        # stderr 重定向到日志文件（非 DEVNULL），便于排查标签/回退/性能问题
        _stderr_log = os.path.join(cwd, "engine_stderr.log")
        self._stderr_fd = open(_stderr_log, "a", encoding="utf-8", buffering=1)
        self.ret = subprocess.Popen(  # 打开管道
            cmds,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fd,   # 写入日志文件而非丢弃
            startupinfo=startupinfo,  # 开启静默模式
        )

        # ── 唯一 stdout 读者（生命周期 = 子进程）──────────────────────────
        # ⚠️ 2026-07-20 卡死根因（中文/韩文全语言）：
        #   旧版「init 专用 reader 线程」在 init 成功后**继续**从 stdout.readline()，
        #   而 runDict 又自己 readline() → 两读者抢同一管道。
        #   结果行常被 init 线程偷走放进废弃队列 → runDict 永久阻塞 →
        #   UI 读条永不结束、强制终止无效（msnTask 卡在 readline，stop 检查不到）。
        # 修复：全生命周期只用一个 reader → Queue；init 与 runDict 都从队列取行。
        self._out_q = queue.Queue()

        def _stdout_reader():
            try:
                while True:
                    if not self.ret or not self.ret.stdout:
                        break
                    raw = self.ret.stdout.readline()
                    if not raw:
                        break  # EOF
                    try:
                        line = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        line = ""
                    self._out_q.put(line)
            except Exception:
                pass
            finally:
                # 哨兵：通知所有等待方管道已关
                try:
                    self._out_q.put(None)
                except Exception:
                    pass

        self._reader_thread = threading.Thread(target=_stdout_reader, daemon=True)
        self._reader_thread.start()

        # 等 "OCR init completed."（带超时；超时 kill 子进程让 Api.start 返回）
        deadline = time.time() + _INIT_TIMEOUT
        while True:
            if self.ret.poll() is not None:  # 子进程已退出
                raise Exception("OCR init fail.")
            remain = deadline - time.time()
            if remain <= 0:
                self.exit()
                raise Exception(
                    f"OCR init 超时（>{_INIT_TIMEOUT}s），引擎可能卡死。"
                    f"建议检查 engine_stderr.log 或重启引擎。"
                )
            try:
                initStr = self._out_q.get(timeout=min(1.0, remain))
            except queue.Empty:
                continue
            if initStr is None:
                raise Exception("OCR init fail.")
            if "OCR init completed." in initStr:
                break
            if "OCR clipboard enbaled." in initStr:  # 剪贴板已启用（兼容旧协议拼写）
                self.__ENABLE_CLIPBOARD = True
        atexit.register(self.exit)  # 注册程序终止时执行强制停止子进程

    def isClipboardEnabled(self) -> bool:
        return self.__ENABLE_CLIPBOARD

    def getRunningMode(self) -> str:
        # 默认管道模式只能运行在本地
        return "local"

    def _read_line(self, timeout_sec):
        """从唯一 stdout 队列取一行；超时返回 None；EOF 返回 ''。"""
        if self._out_q is None:
            return None
        try:
            line = self._out_q.get(timeout=timeout_sec)
        except queue.Empty:
            return None
        if line is None:
            return ""  # EOF 哨兵
        return line

    def runDict(self, writeDict: dict):
        """传入指令字典，发送给引擎进程。\n
        `writeDict`: 指令字典。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        # 检查子进程
        if not self.ret:
            return {"code": 901, "data": f"引擎实例不存在。"}
        if self.ret.poll() is not None:
            return {"code": 902, "data": f"子进程已崩溃。"}
        # 输入信息
        writeStr = jsonDumps(writeDict, ensure_ascii=True, indent=None) + "\n"
        try:
            self.ret.stdin.write(writeStr.encode("utf-8"))
            self.ret.stdin.flush()
        except Exception as e:
            return {
                "code": 902,
                "data": f"向识别器进程传入指令失败，疑似子进程已崩溃。{e}",
            }
        # 从唯一队列读结果（带超时）。超时必须 kill 引擎，否则下次请求仍卡死，
        # 且 Umi-OCR 的 stop 只在 msnTask 返回后检查——不返回就永远停不了。
        getStr = self._read_line(_READ_TIMEOUT)
        if getStr is None:
            # 超时：强制结束引擎，让上层能结束任务、允许用户重试
            try:
                self.exit()
            except Exception:
                pass
            return {
                "code": 902,
                "data": (
                    f"识别超时（>{_READ_TIMEOUT}s），引擎可能卡死，已强制终止引擎进程。"
                    f"请重试；仍失败请查看 engine_stderr.log。"
                ),
            }
        if getStr == "":
            return {"code": 902, "data": "子进程已退出（stdout EOF），无识别结果。"}
        try:
            return jsonLoads(getStr)
        except Exception as e:
            return {
                "code": 904,
                "data": f"识别器输出值反序列化JSON失败。异常信息：[{e}]。原始内容：[{getStr}]",
            }

    def run(self, imgPath: str):
        """对一张本地图片进行文字识别。\n
        `imgPath`: 图片路径。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_path": imgPath}
        return self.runDict(writeDict)

    def runClipboard(self):
        """立刻对剪贴板第一位的图片进行文字识别。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        if self.__ENABLE_CLIPBOARD:
            return self.run("clipboard")
        else:
            raise Exception("剪贴板功能不存在或已禁用。")

    def runBase64(self, imageBase64: str):
        """对一张编码为base64字符串的图片进行文字识别。\n
        `imageBase64`: 图片base64字符串。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_base64": imageBase64}
        return self.runDict(writeDict)

    def runBytes(self, imageBytes):
        """对一张图片的字节流信息进行文字识别。\n
        `imageBytes`: 图片字节流。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        imageBase64 = b64encode(imageBytes).decode("utf-8")
        return self.runBase64(imageBase64)

    def exit(self):
        """关闭引擎子进程（增强版：关管道→优雅终止→强杀→tree-kill）。

        铁律：本方法必须**永不抛异常**到 Umi 宿主。
        语言切换时 Api.start() 会先 stop()；若此处因 stdout 编码（cp1252）
        或 taskkill 失败而抛错，msnPreTask 整段崩 → HTTP 第二次请求永久挂起。
        """
        proc = getattr(self, "ret", None)
        self.ret = None  # 先摘掉引用，避免重入
        if proc is not None:
            try:
                # 1) 先关闭管道，解除可能的 I/O 阻塞（引擎可能卡在 stdin.read）
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                # 2) 优雅终止
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    pass
                # 3) 强杀
                try:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=3)
                except Exception:
                    pass
                # 4) Windows tree-kill（孙子进程）
                if "win32" in str(sysPlatform).lower():
                    try:
                        import subprocess as _sp
                        _pid = getattr(proc, "pid", None)
                        if _pid:
                            _sp.run(
                                ["taskkill", "/F", "/T", "/PID", str(_pid)],
                                capture_output=True,
                                timeout=5,
                                creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
                            )
                    except Exception:
                        pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # 关闭 stderr 日志 fd
        fd = getattr(self, "_stderr_fd", None)
        self._stderr_fd = None
        if fd is not None:
            try:
                fd.close()
            except Exception:
                pass
        try:
            atexit.unregister(self.exit)
        except Exception:
            pass
        # 禁止 print 任何可能非 ASCII 的内容（宿主 stdout 常为 cp1252）

    @staticmethod
    def printResult(res: dict):
        """用于调试，格式化打印识别结果。\n
        `res`: OCR识别结果。"""
        # 识别成功
        if res["code"] == 100:
            index = 1
            for line in res["data"]:
                print(
                    f"{index}-置信度：{round(line['score'], 2)}，文本：{line['text']}"
                )
                index += 1
        elif res["code"] == 100:
            print("图片中未识别出文字。")
        else:
            print(f"图片识别失败。错误码：{res['code']}，错误信息：{res['data']}")

    def __del__(self):
        self.exit()
