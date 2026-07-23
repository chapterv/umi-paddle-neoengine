# -*- coding: utf-8 -*-
"""P1 表结构适配：PaddleX HTML → Umi-OCR 通用 table 协议。"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional


class _TableHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Dict[str, Any]]] = []
        self._row: Optional[List[Dict[str, Any]]] = None
        self._cell: Optional[Dict[str, Any]] = None
        self._table_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            attr = dict(attrs)
            try:
                rowspan = max(1, int(attr.get("rowspan", 1)))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = max(1, int(attr.get("colspan", 1)))
            except (TypeError, ValueError):
                colspan = 1
            self._cell = {
                "text": [],
                "rowspan": rowspan,
                "colspan": colspan,
            }
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append("\n")

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"].append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell["text"]).split())
            self._cell["text"] = text
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1


def html_table_to_cells(html_text: str) -> List[List[str]]:
    """把第一个 HTML table 展开为矩阵；rowspan/colspan 用原文字填充。"""
    if not (html_text or "").strip():
        return []
    parser = _TableHtmlParser()
    parser.feed(html_text)
    raw_rows = parser.rows
    if not raw_rows:
        return []

    occupied: Dict[tuple, str] = {}
    for row_index, row in enumerate(raw_rows):
        col_index = 0
        for cell in row:
            while (row_index, col_index) in occupied:
                col_index += 1
            rowspan = cell["rowspan"]
            colspan = cell["colspan"]
            text = cell["text"]
            for rr in range(row_index, min(len(raw_rows), row_index + rowspan)):
                for cc in range(col_index, col_index + colspan):
                    occupied[(rr, cc)] = text
            col_index += colspan

    n_cols = max((col + 1 for row, col in occupied), default=0)
    return [
        [occupied.get((row, col), "") for col in range(n_cols)]
        for row in range(len(raw_rows))
    ]


def _as_mapping(value: Any) -> Optional[Mapping]:
    if isinstance(value, Mapping):
        return value
    payload = getattr(value, "json", None)
    if payload is not None:
        try:
            payload = payload() if callable(payload) else payload
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, Mapping):
                return payload
        except Exception:
            return None
    return None


def _result_items(output: Any) -> Iterable:
    if output is None:
        return ()
    if _as_mapping(output) is not None:
        return (output,)
    if isinstance(output, (str, bytes)):
        return ()
    try:
        return iter(output)
    except TypeError:
        return (output,)


def _json_safe(value: Any) -> Any:
    """递归转换 NumPy/Paddle 容器，保证引擎结果可直接 JSON 序列化。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(value)


def structure_output_to_table(output: Any) -> Optional[Dict[str, Any]]:
    """提取 PaddleOCR 3.x ``table_res_list[].pred_html``。"""
    for item in _result_items(output):
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        wrapped = _as_mapping(mapping.get("res"))
        if wrapped is not None:
            mapping = wrapped
        table_list = mapping.get("table_res_list") or []
        for table_res in table_list:
            table_mapping = _as_mapping(table_res)
            if table_mapping is None:
                continue
            pred_html = str(table_mapping.get("pred_html") or "")
            cells = html_table_to_cells(pred_html)
            if not cells:
                continue
            return {
                "n_rows": len(cells),
                "n_cols": max((len(row) for row in cells), default=0),
                "cells": cells,
                "source": "structure",
                "html": pred_html,
                "cell_boxes": _json_safe(
                    table_mapping.get("cell_box_list") or []
                ),
            }
    return None


def attach_table_result(
    ocr_result: Dict[str, Any],
    structure_table: Optional[Dict[str, Any]],
    *,
    geometry_builder: Optional[Callable[[List[dict]], Dict[str, Any]]] = None,
    structure_error: str = "",
) -> Dict[str, Any]:
    """结构优先、几何回退；表格失败不得破坏已经成功的 OCR。"""
    if not isinstance(ocr_result, dict) or ocr_result.get("code") != 100:
        return ocr_result
    if structure_table and structure_table.get("cells"):
        ocr_result["table"] = structure_table
        return ocr_result

    errors = []
    if structure_error:
        errors.append(str(structure_error))
    if geometry_builder is not None:
        try:
            table = dict(geometry_builder(ocr_result.get("data") or []))
            table["source"] = "geometry_fallback"
            if structure_error:
                table["structure_error"] = str(structure_error)
            ocr_result["table"] = table
            return ocr_result
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        ocr_result["table_error"] = " | ".join(errors)
    return ocr_result
