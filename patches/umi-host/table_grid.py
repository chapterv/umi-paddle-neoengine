# -*- coding: utf-8 -*-
"""几何表格网格：OCR textBlocks → 二维 cells（无模型，P0）。

参考 #1058 与社区增强版：Y 分行、行内近距合并、X 列聚类、最近列填格。
当 OCR 把整行识别成一块时，按空白/数字再切列（对齐用户样张）。

对齐原则（用户反馈 / 样章-表格.png）：
- 真表是「日期为行、字段为列」的关系表，不是「日期为列」的宽表
- 几何多列已对齐时优先保留，勿用「左对齐文本再切」打乱列对应
- 若误成宽表（首行多日期），自动转置回关系表
- 关系表去掉「数据行全空」的碎列，消除截图空白错位
- 纯数值行补齐左侧标签列；标签独占行与下一行数值合并
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

# 行内：数字/金额序列切分
_NUM_SPLIT = re.compile(
    r"(?<=\d)\s+(?=\d)|(?<=\d)\s+(?=\d)|"  # 数 数
    r"\s{2,}|[\t|｜]+"  # 多空格 / 制表 / 竖线
)
_WS_SPLIT = re.compile(r"\s+")
# 日期 / 金额 / 整数 / 小数
_NUM_OR_DATE = re.compile(
    r"^[\d]+(?:[./\-年]\d+){0,3}日?$"
    r"|^[\d]+(?:\.\d+)?$"
    r"|^[\d,]+\.\d{2}$"
)
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
# 真日期：2020/5/17、2020-05-17、2020年5月17日
_DATE_TOKEN = re.compile(
    r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}$"
    r"|^\d{4}年\d{1,2}月\d{1,2}日?$"
)


def _bbox(tb: dict) -> Tuple[float, float, float, float]:
    """返回 (x0, y0, x1, y1)。优先 normalized_bbox。"""
    if "normalized_bbox" in tb:
        b = tb["normalized_bbox"]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    box = tb.get("box") or [[0, 0], [0, 0], [0, 0], [0, 0]]
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _cx_cy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def _progressive_cluster_values(sorted_vals: List[float], threshold: float) -> List[float]:
    """对已排序 1D 值做递进聚类，返回每簇中心。"""
    if not sorted_vals:
        return []
    clusters: List[List[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _is_date_token(s: str) -> bool:
    t = (s or "").strip()
    return bool(t and _DATE_TOKEN.match(t))


def _is_num_or_date_token(s: str) -> bool:
    t = (s or "").strip().replace(",", "")
    if not t:
        return False
    if _is_date_token(t):
        return True
    if _NUM_OR_DATE.match(t):
        return True
    # 2020/5/17 宽松
    if re.match(r"^\d{2,4}[/\-.\d]+$", t):
        return True
    return False


def _nonempty(row: List[str]) -> List[str]:
    return [c.strip() for c in row if (c or "").strip()]


def _date_row_indices(cells: List[List[str]]) -> List[int]:
    """首格为日期的数据行下标。"""
    out = []
    for i, row in enumerate(cells):
        if row and _is_date_token(row[0]):
            out.append(i)
    return out


def _looks_like_record_table(cells: List[List[str]]) -> bool:
    """关系表：多行以日期开头（样章-表格 正确方向）。"""
    return len(_date_row_indices(cells)) >= 3


def _looks_like_wide_date_table(cells: List[List[str]]) -> bool:
    """
    宽表误排：某一行含 ≥3 个日期（日期在横轴），且多行首格是中文标签。
    对应用户 0750_table.csv 错误方向；应转置为关系表。
    """
    if not cells or len(cells) < 4:
        return False
    if _looks_like_record_table(cells):
        return False
    max_dates = 0
    for row in cells:
        max_dates = max(max_dates, sum(1 for c in row if _is_date_token(c)))
    if max_dates < 3:
        return False
    labelish = 0
    for row in cells:
        if not row:
            continue
        t = (row[0] or "").strip()
        if t and _HAS_CJK.search(t) and not _is_date_token(t):
            labelish += 1
    return labelish >= 3


def _transpose_cells(cells: List[List[str]]) -> List[List[str]]:
    if not cells:
        return cells
    n_cols = max(len(r) for r in cells)
    rows = [_pad_row(r, n_cols) for r in cells]
    n_rows = len(rows)
    return [[rows[r][c] for r in range(n_rows)] for c in range(n_cols)]


def _join_cell(a: str, b: str) -> str:
    """合并两格：尽量保留双方文本（不丢字段）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a == b or a in b:
        return b
    if b in a:
        return a
    # 中文标签 + 数值：标签在前
    if _HAS_CJK.search(a) and _is_num_or_date_token(b):
        return f"{a}{b}" if not a[-1].isspace() else f"{a}{b}"
    if _is_num_or_date_token(a) and _HAS_CJK.search(b):
        return f"{a}{b}"
    if ord(a[-1]) < 128 and ord(b[0]) < 128:
        return f"{a} {b}"
    return a + b


# 已知表头/字段名（长词优先），用于拆 入库金额入库数量 等粘连
_FIELD_LABELS: List[str] = sorted(
    [
        "出入库管理明细表",
        "查询产品日期",
        "查询产品",
        "产品名称",
        "规格型号",
        "入库明细",
        "出库明细",
        "入库数量",
        "入库单价",
        "入库金额",
        "出库数量",
        "出库单价",
        "出库金额",
        "库存数量",
        "库存明细",
        "吾爱破解论坛",
        "备注",
        "日期",
        "单位",
    ],
    key=len,
    reverse=True,
)


def _split_glued_amount_label(text: str) -> List[str]:
    """
    拆开 OCR 粘连：945.00出库数量 / 800.00出库金额 / 15.00入库
    返回 1 或 2 个 token。
    """
    t = (text or "").strip()
    if not t:
        return []
    m = re.match(
        r"^([\d]{1,3}(?:,\d{3})*(?:\.\d+)?|[\d]+(?:\.\d+)?)\s*"
        r"([\u4e00-\u9fff].+)$",
        t,
    )
    if m:
        return [m.group(1), m.group(2)]
    return [t]


def _split_cjk_field_labels(text: str) -> List[str]:
    """
    拆中文字段粘连（对齐 OCR3 独立格）：
    入库金额入库数量 → 入库金额, 入库数量
    产品名称产品1 → 产品名称, 产品1
    单位个个个 → 单位, 个
    """
    t = (text or "").strip()
    if not t:
        return []
    if not _HAS_CJK.search(t):
        return [t]

    m = re.match(r"^(产品名称)(产品\d+)$", t)
    if m:
        return [m.group(1), m.group(2)]
    m = re.match(r"^(规格型号)(规格\d+)$", t)
    if m:
        return [m.group(1), m.group(2)]
    m = re.match(r"^(单位)(个+)$", t)
    if m:
        return [m.group(1), m.group(2)]

    # 入库数量规格型号规格1
    m = re.match(r"^(入库数量|出库数量)(规格型号)(规格\d+)$", t)
    if m:
        return [m.group(1), m.group(2), m.group(3)]

    parts: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        hit = None
        for lab in _FIELD_LABELS:
            if t.startswith(lab, i):
                hit = lab
                break
        if hit:
            parts.append(hit)
            i += len(hit)
            continue
        m2 = re.match(r"(产品\d+|规格\d+|个+)", t[i:])
        if m2:
            tok = m2.group(1)
            parts.append(tok)
            i += len(tok)
            continue
        j = i + 1
        while j < n:
            if any(t.startswith(lab, j) for lab in _FIELD_LABELS):
                break
            if re.match(r"(产品\d+|规格\d+)", t[j:]):
                break
            j += 1
        chunk = t[i:j]
        if chunk:
            parts.append(chunk)
        i = j
    return parts if parts else [t]


@lru_cache(maxsize=512)
def _split_glued_tokens_cached(t: str) -> Tuple[str, ...]:
    """纯字符串拆分的有界缓存；tuple 防止调用方修改缓存值。"""
    if not t:
        return ("",)
    am = _split_glued_amount_label(t)
    if len(am) >= 2:
        return tuple([am[0]] + _split_cjk_field_labels(am[1]))
    return tuple(_split_cjk_field_labels(t))


def _split_glued_tokens(text: str) -> List[str]:
    """金额粘连 + 中文字段粘连的统一入口。"""
    return list(_split_glued_tokens_cached((text or "").strip()))


def _max_reasonable_columns(base_cols: int) -> int:
    """safe split 的统一列增长上限。"""
    return max(base_cols + 4, (base_cols * 3 + 1) // 2)


def _expand_glued_cells(cells: List[List[str]]) -> List[List[str]]:
    """
    粘连单元格拆列（金额+标签、字段名粘连）。
    整表同一列位置插入，避免只扩一行导致错列。
    """
    if not cells:
        return cells
    n_cols = max(len(r) for r in cells)
    rows = [_pad_row(list(r), n_cols) for r in cells]
    split_plan: List[Tuple[List[Tuple[str, ...]], int, int]] = []
    for c in range(n_cols):
        splits = [
            _split_glued_tokens_cached((row[c] or "").strip()) for row in rows
        ]
        max_parts = max(len(p) for p in splits)
        split_plan.append((splits, max_parts, max_parts - 1))

    growth_budget = _max_reasonable_columns(n_cols) - n_cols
    selected = set()
    for c in sorted(range(n_cols), key=lambda i: (split_plan[i][2], i)):
        growth = split_plan[c][2]
        if growth <= growth_budget:
            selected.add(c)
            growth_budget -= growth

    selected_growth = sum(split_plan[c][2] for c in selected)
    if selected_growth > 4:
        date_idxs = _date_row_indices(rows)
        if date_idxs:
            widths = []
            for r in date_idxs:
                width = 0
                for c in range(n_cols):
                    if c in selected:
                        width += sum(
                            1
                            for part in split_plan[c][0][r]
                            if (part or "").strip()
                        )
                    elif (rows[r][c] or "").strip():
                        width += 1
                widths.append(width)
            projected_signature = (
                len(widths),
                min(widths),
                -(max(widths) - min(widths)),
            )
            if projected_signature <= _date_record_signature(rows):
                return rows

    expanded: List[List[str]] = [[] for _ in rows]
    for c, (splits, max_parts, _) in enumerate(split_plan):
        if c not in selected:
            splits = [[row[c]] for row in rows]
            max_parts = 1
        for r, parts in enumerate(splits):
            expanded[r].extend(parts)
            expanded[r].extend([""] * (max_parts - len(parts)))
    return expanded


def score_table(cells: List[List[str]]) -> int:
    """
    通用表格质量分。
    用于在「纯几何」与「文本再切+转置」之间择优。

    不奖励特定样章字段或列数；只观察方向、记录完整度、粘连和
    未切开的数字 blob。
    """
    if not cells:
        return -10**6
    score = 0
    if _looks_like_wide_date_table(cells):
        score -= 100
    flat_cells = [(c or "").strip() for r in cells for c in r]
    for row in cells:
        nonempty = sum(1 for cell in row if (cell or "").strip())
        if nonempty >= 2:
            score += min(12, nonempty)
    for c in flat_cells:
        if c and len(_split_glued_tokens_cached(c)) >= 2:
            score -= 18
        if re.search(r"\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?){2,}", c):
            score -= 45
    return score


def _table_density(cells: List[List[str]]) -> float:
    """候选同分时优先内容更集中、空洞更少的矩阵。"""
    if not cells:
        return 0.0
    n_cols = max((len(r) for r in cells), default=0)
    if n_cols == 0:
        return 0.0
    nonempty = sum(1 for row in cells for cell in row if (cell or "").strip())
    return nonempty / (len(cells) * n_cols)


def _is_high_quality_record(cells: List[List[str]]) -> bool:
    """OCR3 级：关系表 + 标准表头 + 数据行产品名干净。"""
    if not _looks_like_record_table(cells):
        return False
    flat = " ".join((c or "") for r in cells for c in r)
    if "日期" not in flat or "产品名称" not in flat:
        return False
    # 快速保护只能保护已经没有确定性粘连的表。汇总/表头中的
    # ``945.00出库数量`` 与数据行粘连同样会破坏结构，不能被忽略。
    if any(
        len(_split_glued_tokens_cached((cell or "").strip())) >= 2
        for row in cells
        for cell in row
        if (cell or "").strip()
    ):
        return False
    bad = 0
    for i in _date_row_indices(cells):
        r = cells[i]
        if any(
            re.search(r"产品名称产品|规格型号规格|个个个|入库金额入库", c or "")
            for c in r
        ):
            bad += 1
    return bad == 0 and score_table(cells) >= 100


def _col_has_content(rows: List[List[str]], c: int, idxs: Sequence[int]) -> bool:
    return any((rows[i][c] or "").strip() for i in idxs if i < len(rows) and c < len(rows[i]))


def _should_preserve_sparse_column(texts: List[str]) -> bool:
    """
    数据行虽空、但非数据行有「不可丢」内容的列必须保留：
    汇总金额、备注、页脚/水印 URL、带数字的粘连串等。
    纯表头标签（入库单价）可并入邻列以消空白。
    """
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if "备注" in t:
            return True
        if re.search(r"https?://|www\.|\.com|\.cn|\.net", t, re.I):
            return True
        if re.search(r"52pojie|吾爱", t, re.I):
            return True
        # 含金额/整数汇总（945.00、800、13）
        if re.search(r"\d+\.\d{2}", t) or re.match(r"^\d{2,}$", t):
            return True
        parts = _split_glued_amount_label(t)
        if len(parts) >= 2:
            return True
        # 标题类整行
        if any(k in t for k in ("明细表", "管理", "合计", "总计")):
            return True
    return False


def densify_record_columns(cells: List[List[str]]) -> List[List[str]]:
    """
    关系表去碎列（最优解：紧凑 + 不丢字段）。

    仅把无数据列中的普通表头对齐到邻近数据列；汇总、备注、水印和
    带数字的粘连内容保持在原列。全表空列随后才会删除。
    """
    if not cells:
        return cells
    date_idxs = _date_row_indices(cells)
    if len(date_idxs) < 3:
        return densify_cells(cells)

    n_cols = max(len(r) for r in cells)
    rows = [_pad_row(list(r), n_cols) for r in cells]
    date_set = set(date_idxs)
    non_date = [i for i in range(len(rows)) if i not in date_set]

    def has_data(c: int) -> bool:
        return _col_has_content(rows, c, date_idxs)

    def has_any(c: int) -> bool:
        return _col_has_content(rows, c, range(len(rows)))

    for c in range(n_cols):
        if has_data(c) or not has_any(c):
            continue
        texts = [
            (rows[r][c] or "").strip()
            for r in non_date
            if c < len(rows[r]) and (rows[r][c] or "").strip()
        ]
        if _should_preserve_sparse_column(texts):
            continue
        target = None
        for d in range(1, n_cols):
            if c + d < n_cols and has_data(c + d):
                target = c + d
                break
            if c - d >= 0 and has_data(c - d):
                target = c - d
                break
        if target is None:
            continue
        source_rows = [
            r for r in range(len(rows)) if (rows[r][c] or "").strip()
        ]
        # 整列移动必须逐格无碰撞；任何目标已占用就保留原列。
        if any((rows[r][target] or "").strip() for r in source_rows):
            continue
        for r in source_rows:
            rows[r][target] = (rows[r][c] or "").strip()
            rows[r][c] = ""

    return densify_cells(rows)


def _drop_blank_rows(cells: List[List[str]]) -> List[List[str]]:
    return [r for r in cells if any((c or "").strip() for c in r)]


def _character_multiset(cells: List[List[str]]) -> Counter:
    """忽略排版空白后的字符多重集，用作内容守恒硬门槛。"""
    return Counter(
        ch
        for row in cells
        for cell in row
        for ch in (cell or "")
        if not ch.isspace()
    )


def _date_record_signature(cells: List[List[str]]) -> Tuple[int, int, int]:
    """日期记录数、最小完整度、行宽一致性；仅用于拒绝爆列候选。"""
    date_rows = [cells[i] for i in _date_row_indices(cells)]
    if not date_rows:
        return (0, 0, 0)
    widths = [
        sum(1 for cell in row if (cell or "").strip()) for row in date_rows
    ]
    return (len(date_rows), min(widths), -(max(widths) - min(widths)))


def _bounded_table_candidate(
    base: List[List[str]], candidate: List[List[str]]
) -> List[List[str]]:
    """拒绝内容损失，及没有记录完整度收益的横向爆列候选。"""
    compact_base = densify_cells(base)
    if _character_multiset(candidate) != _character_multiset(base):
        return compact_base
    base_cols = max((len(row) for row in compact_base), default=0)
    candidate_cols = max((len(row) for row in candidate), default=0)
    max_reasonable_cols = _max_reasonable_columns(base_cols)
    base_has_dates = bool(_date_row_indices(compact_base))
    if (
        (
            candidate_cols > max_reasonable_cols
            or (base_has_dates and candidate_cols > base_cols + 4)
        )
        and _date_record_signature(candidate) <= _date_record_signature(compact_base)
    ):
        return compact_base
    return candidate


def _orient_and_compact(cells: List[List[str]]) -> List[List[str]]:
    """方向纠正 + 粘连拆分 + 去碎列；用 score 避免越整理越差。"""
    if not cells:
        return cells
    base = [list(r) for r in cells]
    s0 = score_table(base)

    # OCR3 级：几乎不动
    if _is_high_quality_record(base):
        return densify_cells(base)

    work = _expand_glued_cells(base)
    if _looks_like_wide_date_table(work):
        tr = densify_cells(_transpose_cells(work))
        # 仅当转置后质量不更差才采用
        if score_table(tr) >= s0 - 10:
            work = tr
        else:
            work = densify_cells(work)
    if _looks_like_record_table(work):
        work = densify_record_columns(work)
    else:
        work = densify_cells(work)
    work = _drop_blank_rows(work)
    # 择优：整理后若明显更差则回退
    if score_table(work) + 5 < s0 and s0 >= 80:
        return densify_cells(base)
    return _bounded_table_candidate(base, work)


def _is_values_only(row: List[str]) -> bool:
    ne = _nonempty(row)
    if len(ne) < 2:
        # 单格也可能是纯数值行（如 945.00）
        return len(ne) == 1 and _is_num_or_date_token(ne[0])
    return all(_is_num_or_date_token(c) for c in ne)


def _is_label_only(row: List[str]) -> bool:
    """行内容主要是中文标签、几乎没有「多列数值」。"""
    ne = _nonempty(row)
    if not ne:
        return False
    if _is_values_only(row):
        return False
    joined = "".join(ne)
    if not _HAS_CJK.search(joined):
        return False
    # 单格中文/标题，或仅 1～2 个非数值 token
    num_toks = sum(1 for c in ne if _is_num_or_date_token(c))
    if num_toks == 0 and len(ne) <= 3:
        return True
    # 「入库明细入库单价」一类：全中文挤在一格
    if len(ne) == 1 and _HAS_CJK.search(ne[0]) and not re.search(r"\d{2,}", ne[0]):
        return True
    return False


def _merge_row_items(
    items: List[dict],
    merge_ratio: float = 0.35,
    col_anchors: Sequence[float] | None = None,
) -> List[dict]:
    """同一行内，水平间隙很小的块合并文本（增强版思路）。"""
    if len(items) < 2:
        return items
    items = sorted(items, key=lambda d: d["cx"])
    def anchor_index(x: float) -> int | None:
        if not col_anchors:
            return None
        return min(range(len(col_anchors)), key=lambda i: abs(col_anchors[i] - x))

    first = items[0].copy()
    first["anchor_i"] = anchor_index(first["cx"])
    first["merge_w"] = first["w"]
    merged = [first]
    for pb in items[1:]:
        cur = merged[-1]
        gap = pb["x0"] - cur["x1"]
        pb_anchor = anchor_index(pb["cx"])
        # 多行数据推断出的列锚不同，说明这是相邻列而不是同格 OCR 片段。
        # 同时以原始块宽而不是累计后的 cur.w 计算阈值，避免连锁吞并。
        ref_w = max(cur.get("merge_w", cur["w"]), pb["w"], 1.0)
        same_anchor = (
            cur.get("anchor_i") is None
            or pb_anchor is None
            or cur.get("anchor_i") == pb_anchor
        )
        if same_anchor and gap < ref_w * merge_ratio:
            # 拼接
            t1, t2 = cur["text"], pb["text"]
            if t1 and t2 and ord(t1[-1]) < 128 and ord(t2[0]) < 128:
                cur["text"] = t1 + " " + t2
            else:
                cur["text"] = t1 + t2
            cur["x1"] = max(cur["x1"], pb["x1"])
            cur["cx"] = (cur["x0"] + cur["x1"]) * 0.5
            cur["w"] = cur["x1"] - cur["x0"]
            cur["merge_w"] = ref_w
            cur["idxs"].extend(pb["idxs"])
        else:
            nxt = pb.copy()
            nxt["anchor_i"] = pb_anchor
            nxt["merge_w"] = nxt["w"]
            merged.append(nxt)
    return merged


def _split_line_to_cells(text: str) -> List[str]:
    """整行文本切列：多空格/竖线/中文与数字交界/数字序列/单空格。"""
    text = (text or "").strip()
    if not text:
        return [""]
    # 先按多空格或 | 切
    if re.search(r"\s{2,}|[\t|｜]", text):
        parts = re.split(r"\s{2,}|\t+|\|+|｜+", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts
    # 中文/标签 与 日期数字 之间切开：查询产品日期 2020/5/17 …
    if re.search(r"[\u4e00-\u9fff]\s+[\d]", text) or re.search(r"\d\s+\d", text):
        parts = re.split(
            r"(?<=[\u4e00-\u9fff])\s+(?=[\d])|(?<=[\d./])\s+(?=[\d./])",
            text,
        )
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts
    # 中文标签后紧跟数字串：尝试按单个空格切（多 token）
    parts = _WS_SPLIT.split(text)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts
    return [text]


def _row_as_one_string(row: List[str]) -> str:
    nonempty = [c for c in row if (c or "").strip()]
    if len(nonempty) == 1:
        return nonempty[0].strip()
    # 内容几乎全在第一列、其余空：仍视为「整行一坨」
    if row and (row[0] or "").strip():
        rest = [c for c in row[1:] if (c or "").strip()]
        if not rest and len(_split_line_to_cells(row[0])) >= 2:
            return row[0].strip()
    return ""


def densify_cells(cells: List[List[str]]) -> List[List[str]]:
    """删除整列皆空的列，消除截图/TSV 错位空列。"""
    if not cells:
        return cells
    n_cols = max(len(r) for r in cells)
    if n_cols == 0:
        return cells
    rows = [list(r) + [""] * (n_cols - len(r)) for r in cells]
    keep = [
        c
        for c in range(n_cols)
        if any((rows[r][c] or "").strip() for r in range(len(rows)))
    ]
    if not keep:
        return [[""] for _ in rows]
    return [[rows[r][c] for c in keep] for r in range(len(rows))]


def _pad_row(row: List[str], n_cols: int) -> List[str]:
    r = list(row)
    if len(r) < n_cols:
        r = r + [""] * (n_cols - len(r))
    return r[:n_cols]


def _geometry_is_stable_multi_col(cells: List[List[str]]) -> bool:
    """几何结果已有多列且多行有 ≥2 非空格 → 勿再文本左对齐切列（会错列）。"""
    if not cells:
        return False
    n_cols = max(len(r) for r in cells)
    if n_cols < 2:
        return False
    multi = 0
    for row in cells:
        if sum(1 for c in row if (c or "").strip()) >= 2:
            multi += 1
    # 至少约 1/3 行是多列，或至少 2 行多列
    return multi >= 2 or multi >= max(1, (len(cells) + 2) // 3)


def _align_value_rows(cells: List[List[str]]) -> List[List[str]]:
    """
    纯数值行左对齐会与「标签+数值」表头错列。
    仅当表中已存在「标签列 + 数据列」时，把纯数值行左侧补一个空标签格。
    绝不截断已有列。
    """
    if not cells:
        return cells

    # 是否存在「首格非数值 + 后续含数值」→ 确定有标签列
    has_label_col = False
    labeled_data_widths: List[int] = []
    value_only_widths: List[int] = []
    for row in cells:
        ne = _nonempty(row)
        if len(ne) < 2:
            continue
        if not _is_num_or_date_token(ne[0]) and any(
            _is_num_or_date_token(c) for c in ne[1:]
        ):
            has_label_col = True
            labeled_data_widths.append(len(ne) - 1)
        elif _is_values_only(row):
            value_only_widths.append(len(ne))

    n_cols = max(len(r) for r in cells)
    max_ne = max((len(_nonempty(r)) for r in cells), default=0)
    if max_ne < 2:
        return [_pad_row(r, n_cols) for r in cells]

    # 无标签列形态：整表都是数值网格 → 只统一 pad，不左移
    if not has_label_col:
        return [_pad_row(list(r), n_cols) for r in cells]

    target_data = max(labeled_data_widths) if labeled_data_widths else max(
        value_only_widths or [max_ne - 1]
    )
    # 目标列 = 标签 + 数据；且不小于当前最大列数
    target_cols = max(n_cols, target_data + 1)

    out: List[List[str]] = []
    for row in cells:
        ne = _nonempty(row)
        if not ne:
            out.append(_pad_row(row, target_cols))
            continue
        # 纯数值行且数量≈数据列宽 → 左侧补空标签（扩展而非截断）
        if _is_values_only(row) and len(ne) >= 2:
            if len(ne) <= target_data:
                padded = [""] + ne
                out.append(_pad_row(padded, max(target_cols, len(padded))))
                continue
            # 已比数据列更宽：可能已含空标签，原样 pad
        out.append(_pad_row(list(row), target_cols))
    return out


def _merge_label_value_rows(cells: List[List[str]]) -> List[List[str]]:
    """标签独占行 + 下一行纯数值 → 合成一行，消除「标题与字段分离」。"""
    if not cells:
        return cells
    out: List[List[str]] = []
    i = 0
    n = len(cells)
    while i < n:
        row = cells[i]
        if i + 1 < n and _is_label_only(row) and _is_values_only(cells[i + 1]):
            label_ne = _nonempty(row)
            vals = _nonempty(cells[i + 1])
            # 独立标题须保留边界；空串拼接会制造新的粘连字段。
            label = label_ne[0] if len(label_ne) == 1 else " ".join(label_ne)
            # 若数值行已带空标签位
            if vals and vals[0] == "":
                merged = [label] + vals[1:]
            elif _is_values_only(cells[i + 1]):
                merged = [label] + vals
            else:
                merged = [label] + vals
            out.append(merged)
            i += 2
            continue
        # 单格纯数值孤立、上一行已有标签+值：保留（合计行）
        out.append(list(row))
        i += 1
    return out


def _split_leading_num_from_label(cells: List[List[str]]) -> List[List[str]]:
    """
    「26 出库数量」/「800.00 出库金额」：首格数字粘在标签上。
    若本行后续已是数值列，把粘连数字挪到独立逻辑不破坏列；
    仅拆首格内部：num + label → 若像「合计残留」则 label 保留在 col0、num 单独不成列。
    策略：首格匹配 ^数字 + 中文标签$ 且后继为数值 → 把数字和标签
    保留为相邻单元格。默认整理不得用“疑似上行合计”作理由丢弃数字。
    """
    out: List[List[str]] = []
    for row in cells:
        if not row:
            out.append(row)
            continue
        r = list(row)
        head = (r[0] or "").strip()
        m = re.match(
            r"^([\d]+(?:\.\d+)?)\s+([\u4e00-\u9fff].+)$",
            head,
        )
        if m and len(_nonempty(r)) >= 2:
            # 后继多为数值：拆开而非静默删除数字前缀。
            rest = _nonempty(r[1:])
            if rest and sum(1 for c in rest if _is_num_or_date_token(c)) >= max(1, len(rest) // 2):
                r[0] = m.group(1).strip()
                r.insert(1, m.group(2).strip())
        out.append(r)
    return out


def _refine_columns_by_split(cells: List[List[str]]) -> List[List[str]]:
    """
    几何列过少，或「内容全挤在第一列」，则按空白/数字再切列。
    切完后做数值行对齐，避免左对齐导致错列。
    """
    if not cells:
        return cells

    split_rows: List[List[str]] = []
    need = False
    for row in cells:
        one = _row_as_one_string(row)
        if one:
            parts = _split_line_to_cells(one)
            if len(parts) >= 2:
                need = True
                split_rows.append(parts)
            else:
                split_rows.append(list(row))
        else:
            # 已有多列实质内容：保持几何列，勿整体重切
            refined = list(row)
            for i, cell in enumerate(row):
                if cell and len(_split_line_to_cells(cell)) >= 3 and not any(
                    (row[j] or "").strip() for j in range(len(row)) if j != i
                ):
                    parts = _split_line_to_cells(cell)
                    need = True
                    refined = parts
                    break
            split_rows.append(refined)

    max_tok = max((len(r) for r in split_rows), default=0)
    n_cols = max((len(r) for r in cells), default=0)
    if not need and max_tok <= n_cols:
        return cells
    if max_tok < 2:
        return cells

    n_cols2 = max_tok
    out: List[List[str]] = []
    for parts in split_rows:
        row = list(parts) + [""] * (n_cols2 - len(parts))
        out.append(row[:n_cols2])
    return _align_value_rows(out)


def normalize_table_cells(cells: List[List[str]]) -> List[List[str]]:
    """导出/展示前统一：切列/对齐/合并 + 宽表转置 + 关系表去碎列。"""
    if not cells:
        return []
    # OCR3 级高质量：禁止激进再切/折叠
    if _is_high_quality_record(cells):
        return densify_cells([list(r) for r in cells])

    # 几何多列已稳：轻量对齐 + 定向整理
    if _geometry_is_stable_multi_col(cells):
        cells = densify_cells(cells)
        cells = _align_value_rows(cells)
        cells = _merge_label_value_rows(cells)
        cells = _split_leading_num_from_label(cells)
        return _orient_and_compact(cells)

    cells = _refine_columns_by_split(cells)
    cells = _align_value_rows(cells)
    cells = _merge_label_value_rows(cells)
    cells = _split_leading_num_from_label(cells)
    cells = densify_cells(cells)
    cells = _align_value_rows(cells)
    return _orient_and_compact(cells)


def cells_from_plain_text(text: str) -> List[List[str]]:
    """无 table 结构时，从纯文本行重建二维表（供 CSV 兜底）。"""
    if not (text or "").strip():
        return []
    rows: List[List[str]] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        if "\t" in line:
            rows.append([c.strip() for c in line.split("\t")])
        else:
            rows.append(_split_line_to_cells(line))
    return normalize_table_cells(rows)


def build_table(
    text_blocks: List[dict],
    *,
    row_tol_ratio: float = 0.6,
    col_gap_ratio: float = 0.7,
    merge_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    将 text_blocks 聚类为网格。

    row_tol_ratio: 行聚类阈值 = ratio * 中位字高
    col_gap_ratio: 列递进聚类阈值 = ratio * 中位字高（略松，减少空列错位）
    merge_ratio: 行内近距合并阈值（相对块宽）
    """
    blocks_in = [tb for tb in text_blocks if (tb.get("text") or "").strip()]
    if not blocks_in:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "cells": [],
            "row_ys": [],
            "col_xs": [],
        }

    items: List[dict] = []
    for i, tb in enumerate(blocks_in):
        b = _bbox(tb)
        x0, y0, x1, y1 = b
        cx, cy = _cx_cy(b)
        h = max(1e-3, y1 - y0)
        w = max(1e-3, x1 - x0)
        items.append(
            {
                "tb": tb,
                "idxs": [i],
                "x0": x0,
                "x1": x1,
                "cx": cx,
                "cy": cy,
                "h": h,
                "w": w,
                "text": tb["text"].strip(),
            }
        )

    med_h = _median([it["h"] for it in items])
    row_th = max(med_h * row_tol_ratio, 1.0)

    # ----- 行聚类（按 cy 排序递进）-----
    items_y = sorted(items, key=lambda d: d["cy"])
    raw_rows: List[List[dict]] = [[items_y[0]]]
    for it in items_y[1:]:
        if abs(it["cy"] - raw_rows[-1][-1]["cy"]) <= row_th:
            raw_rows[-1].append(it)
        else:
            raw_rows.append([it])

    # merge 前先从全部行推断列锚。标题块跨不同锚时不能被当作同一单元格拼接。
    raw_col_xs = _progressive_cluster_values(
        sorted(it["cx"] for row in raw_rows for it in row),
        max(med_h * col_gap_ratio, 2.0),
    )
    # 行内近距合并
    rows = [
        _merge_row_items(r, merge_ratio=merge_ratio, col_anchors=raw_col_xs)
        for r in raw_rows
    ]
    row_ys = [_median([it["cy"] for it in r]) for r in rows]
    n_rows = len(rows)

    # ----- 列：优先用 cx（#1058），辅以 x0；阈值略松减少碎列 -----
    all_cx = sorted(it["cx"] for r in rows for it in r)
    gaps = [
        all_cx[i + 1] - all_cx[i]
        for i in range(len(all_cx) - 1)
        if all_cx[i + 1] - all_cx[i] > 1
    ]
    med_gap = _median(gaps) if gaps else med_h
    # 社区增强版约 0.7 * 字高；大间隙时仍保持合理簇
    col_th = max(med_h * col_gap_ratio, 2.0)
    if med_gap > 0:
        # 不要比「中位间隙的 55%」更碎
        col_th = max(col_th, min(med_gap * 0.55, med_h * 1.2))

    col_xs = _progressive_cluster_values(all_cx, col_th)
    if not col_xs:
        col_xs = [all_cx[0]]
    n_cols = len(col_xs)

    # ----- 最近列填格（以 cx 为主，与列锚一致）-----
    cells_geo: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for ri, row in enumerate(rows):
        for it in row:
            nearest = min(
                range(n_cols),
                key=lambda ci: abs(col_xs[ci] - it["cx"]) * 0.75
                + abs(col_xs[ci] - it["x0"]) * 0.25,
            )
            if cells_geo[ri][nearest]:
                t1, t2 = cells_geo[ri][nearest], it["text"]
                if t1 and t2 and ord(t1[-1]) < 128 and ord(t2[0]) < 128:
                    cells_geo[ri][nearest] = t1 + " " + t2
                else:
                    cells_geo[ri][nearest] = t1 + t2
            else:
                cells_geo[ri][nearest] = it["text"]

    # ----- 双路径择优：纯几何(贴近 OCR3) vs 全文 normalize(修宽表/切 blob) -----
    pure = densify_cells([list(r) for r in cells_geo])
    full = normalize_table_cells([list(r) for r in cells_geo])
    pure_score, full_score = score_table(pure), score_table(full)
    if pure_score > full_score:
        cells = pure
    elif full_score > pure_score:
        cells = full
    else:
        # 同分：不再把“列更多”当质量；优先空洞更少的矩阵。
        # 密度也相同时保留纯几何，避免无收益重写。
        cells = full if _table_density(full) > _table_density(pure) else pure
    n_rows = len(cells)
    n_cols = max((len(r) for r in cells), default=0)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cells": cells,
        "row_ys": row_ys,
        "col_xs": col_xs,
    }


def cells_to_tsv(cells: List[List[str]]) -> str:
    lines = []
    for row in cells:
        lines.append("\t".join((c or "").replace("\t", " ") for c in row))
    return "\n".join(lines)


def cells_to_md(cells: List[List[str]]) -> str:
    if not cells:
        return ""
    n_cols = max(len(r) for r in cells)

    def pad(row):
        return list(row) + [""] * (n_cols - len(row))

    rows = [pad(r) for r in cells]
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cells_to_csv_rows(cells: List[List[str]]) -> List[List[str]]:
    return [list(r) for r in cells]
