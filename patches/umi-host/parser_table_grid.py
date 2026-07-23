# 排版解析-表格网格（几何，P0）
# 新增方案 table_grid，不替换任何既有 parser。
#
# 预览：保留 OCR 原始 box（不改造成左上角假框）。
# 结构：self.last_table + 首块 _table 供导出；文本区仍按原块阅读顺序展示。

from .tbpu import Tbpu
from .parser_tools.line_preprocessing import linePreprocessing
from .parser_tools.table_grid import build_table, _bbox


class TableGrid(Tbpu):
    def __init__(self):
        self.tbpuName = "排版解析-表格-几何网格"
        self.last_table = None

    def run(self, textBlocks):
        raw = list(textBlocks) if textBlocks else []
        tbs = linePreprocessing(raw)
        if not tbs:
            self.last_table = {
                "n_rows": 0,
                "n_cols": 0,
                "cells": [],
                "source": "geometry",
            }
            return []

        table = build_table(tbs)
        table["source"] = "geometry"
        self.last_table = table

        # 阅读顺序：上→下、左→右（用真实坐标），保留原始 box 供左侧叠加预览
        def sort_key(tb):
            x0, y0, x1, y1 = _bbox(tb)
            return (y0, x0)

        ordered = sorted(tbs, key=sort_key)
        # 根据 table 行高粗略设 end：同行用 tab 感（空格），行末换行
        # 用 cy 是否接近判断是否同行
        for i, tb in enumerate(ordered):
            if "normalized_bbox" in tb:
                del tb["normalized_bbox"]
            if i + 1 >= len(ordered):
                tb["end"] = "\n"
                continue
            _, y0a, _, y1a = _bbox(tb)
            _, y0b, _, y1b = _bbox(ordered[i + 1])
            cya, cyb = (y0a + y1a) * 0.5, (y0b + y1b) * 0.5
            ha = max(1.0, y1a - y0a)
            if abs(cyb - cya) <= ha * 0.6:
                tb["end"] = " "  # 同行块：空格（预览自然）；结构导出用 table.cells
            else:
                tb["end"] = "\n"

        # 结构表挂在首块，promote 后删除；预览不依赖假框
        if ordered:
            ordered[0]["_table"] = table
            # 文本预览增强：若 UI 拼 data 文本，用户仍看到原 OCR 字；
            # 另在首块 text 前不改写，避免丢坐标语义
        return ordered
