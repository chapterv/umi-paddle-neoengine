# -*- coding: utf-8 -*-
import os, sys


def _ensure_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


MENU = (
    "\n请选择预下载的模型范围（首次识别缺模型也会自动下载）：\n"
    "[1] 完整版   : V4 + V6 + V5多语言 · ONNX + MKL-DNN  ~最大\n"
    "[2] 最小可用 : PP-OCRv6（中文）· 仅 ONNX            ~最小\n"
    "[3] 多语言   : V6 + V5多语言 · 仅 ONNX              ~中等\n"
    "（直接回车 = 选项2）\n"
)


def main():
    _ensure_utf8()
    if len(sys.argv) < 2:
        sys.exit("usage: menu.py <choice_out_file>")
    out = sys.argv[1]
    print(MENU, flush=True)
    try:
        raw = input("请输入 1/2/3（默认 2）: ")
    except EOFError:
        raw = ""
    choice = raw.strip() or "2"
    if choice not in ("1", "2", "3"):
        choice = "2"
    with open(out, "w", encoding="utf-8") as f:
        f.write(choice)
    print("（已选择选项 %s）" % choice, flush=True)


if __name__ == "__main__":
    main()
