# -*- coding: utf-8 -*-
"""P1 公式识别 Spike：结构化公式 region 与混排合并。"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

_FORMULA_HINT_CHARS = set("√πτωξθλμναβγδεζηικρσφψΩ∞∑∫∂±×÷≈≠≤≥∈∉^=")
_FORMULA_EXPR_RE = re.compile(r"[A-Za-z]\s*[\^_=+\-×÷*/<>]\s*[A-Za-z0-9(]")
_FORMULA_TERM_RE = re.compile(r"(sqrt|frac|sum|int|lim|sin|cos|tan|log|ln|alpha|beta)", re.I)


def _as_sequence(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return value


def _is_scalar_number(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        float(value)
        return True
    except Exception:
        return False


def _point_xy(value: Any) -> Optional[List[int]]:
    seq = _as_sequence(value)
    if isinstance(seq, (list, tuple)) and len(seq) >= 2:
        try:
            return [int(round(float(seq[0]))), int(round(float(seq[1])))]
        except Exception:
            return None
    return None


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


def _formula_text_score(text: str) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    score = 0
    if any(ch in _FORMULA_HINT_CHARS for ch in raw):
        score += 2
    if _FORMULA_EXPR_RE.search(raw):
        score += 2
    if _FORMULA_TERM_RE.search(raw):
        score += 2
    has_ascii = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in raw)
    has_digit = any(ch.isdigit() for ch in raw)
    if has_ascii and has_digit and len(raw) <= 24:
        score += 1
    if len(raw) <= 12 and sum(ch in _FORMULA_HINT_CHARS for ch in raw) >= 2:
        score += 1
    return score


def should_run_formula_layout(ocr_blocks: Optional[List[dict]]) -> bool:
    """Cheap gate before loading/running the heavy layout+formula pipeline.

    Goal: when formula recognition is merely enabled as a capability, ordinary
    prose screenshots should stay on the fast OCR-only path.  Explicit
    whole-image formula mode still bypasses this gate.
    """
    blocks = list(ocr_blocks or [])
    if not blocks:
        return False
    total_score = 0
    strong_hits = 0
    short_formulaish = 0
    for block in blocks:
        text = str((block or {}).get("text") or "").strip()
        if not text:
            continue
        score = _formula_text_score(text)
        total_score += score
        if score >= 2:
            strong_hits += 1
        if score >= 1 and len(text) <= 24:
            short_formulaish += 1
    if strong_hits >= 1 and total_score >= 2:
        return True
    if short_formulaish >= 2 and total_score >= 3:
        return True
    return False
    if isinstance(output, (str, bytes)):
        return ()
    try:
        return iter(output)
    except TypeError:
        return (output,)


def _to_quad(value: Any) -> List[List[int]]:
    seq = _as_sequence(value)
    if isinstance(seq, (list, tuple)) and len(seq) == 4:
        points = [_point_xy(item) for item in seq]
        if all(point is not None for point in points):
            return points
    if isinstance(seq, (list, tuple)) and len(seq) == 4 and all(
        _is_scalar_number(item) for item in seq
    ):
        x1, y1, x2, y2 = [int(round(float(v))) for v in seq]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return []


def _rect_key(value: Any) -> tuple[int, int, int, int] | tuple[()]:
    quad = _to_quad(value)
    if quad:
        return tuple(_bbox(quad))
    seq = _as_sequence(value)
    if isinstance(seq, (list, tuple)) and len(seq) == 4 and all(
        _is_scalar_number(item) for item in seq
    ):
        return tuple(int(round(float(v))) for v in seq)
    return ()


def _bbox(quad: List[List[int]]) -> List[int]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _overlap_ratio(quad_a: List[List[int]], quad_b: List[List[int]]) -> float:
    if not quad_a or not quad_b:
        return 0.0
    ax1, ay1, ax2, ay2 = _bbox(quad_a)
    bx1, by1, bx2, by2 = _bbox(quad_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / min(area_a, area_b)


def _region_sort_key(region: Dict[str, Any]) -> tuple[int, int]:
    box = region.get("box") or []
    if not box:
        return (10**9, 10**9)
    rect = _bbox(box)
    return (rect[1], rect[0])


def _region(
    *,
    idx: int,
    kind: str,
    box: List[List[int]],
    text: str = "",
    latex: str = "",
    score: float = 0.0,
    source: str = "",
) -> Dict[str, Any]:
    return {
        "id": f"{kind}-{idx}",
        "type": kind,
        "box": box,
        "text": text,
        "latex": latex,
        "score": float(score or 0.0),
        "source": source,
    }


def _match_best_ocr(
    target_box: List[List[int]],
    ocr_blocks: List[dict],
    used_indices: set[int],
    threshold: float = 0.35,
) -> tuple[Optional[int], Optional[dict]]:
    best_idx = None
    best_block = None
    best_score = threshold
    for idx, block in enumerate(ocr_blocks):
        if idx in used_indices:
            continue
        block_box = block.get("box") or []
        score = _overlap_ratio(target_box, block_box)
        if score > best_score:
            best_idx = idx
            best_block = block
            best_score = score
    return best_idx, best_block


def _formula_mapping_list(output: Any) -> List[Mapping]:
    out = []
    for item in _result_items(output):
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        wrapped = _as_mapping(mapping.get("res"))
        out.append(wrapped if wrapped is not None else mapping)
    return out


def build_formula_payload(
    output: Any,
    *,
    mode: str,
    model_name: str,
    ocr_blocks: Optional[List[dict]] = None,
    image_shape: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    mappings = _formula_mapping_list(output)
    mapping = mappings[0] if mappings else {}
    layout = _as_mapping(mapping.get("layout_det_res")) or {}
    formula_results = list(mapping.get("formula_res_list") or [])
    ocr_blocks = list(ocr_blocks or [])
    regions: List[Dict[str, Any]] = []
    used_ocr: set[int] = set()

    if mode == "whole_image":
        height = int(image_shape[0]) if image_shape else 0
        width = int(image_shape[1]) if image_shape and len(image_shape) > 1 else 0
        full_box = [[0, 0], [width, 0], [width, height], [0, height]]
        for idx, item in enumerate(formula_results, start=1):
            latex = str((item or {}).get("rec_formula") or "")
            regions.append(
                _region(
                    idx=idx,
                    kind="formula",
                    box=full_box,
                    text=latex,
                    latex=latex,
                    score=float((item or {}).get("score") or 0.0),
                    source="formula_recognition",
                )
            )
    else:
        formula_by_rect = {}
        ordered_formula = []
        for item in formula_results:
            latex = str((item or {}).get("rec_formula") or "")
            rect = _rect_key((item or {}).get("dt_polys") or [])
            if len(rect) == 4:
                formula_by_rect[rect] = item
            ordered_formula.append(item)
        next_formula = iter(ordered_formula)

        formula_index = 0
        formula_number_index = 0
        for box in list(layout.get("boxes") or []):
            label = str(box.get("label") or "").lower()
            quad = _to_quad(box.get("coordinate") or [])
            if not quad:
                continue
            if label == "formula":
                formula_index += 1
                rect = _rect_key(box.get("coordinate") or [])
                item = formula_by_rect.get(rect)
                if item is None:
                    item = next(next_formula, {})
                latex = str((item or {}).get("rec_formula") or "")
                matched_idx, _ = _match_best_ocr(quad, ocr_blocks, used_ocr, threshold=0.25)
                if matched_idx is not None:
                    used_ocr.add(matched_idx)
                regions.append(
                    _region(
                        idx=formula_index,
                        kind="formula",
                        box=quad,
                        text=latex,
                        latex=latex,
                        score=float(box.get("score") or 0.0),
                        source="formula_recognition",
                    )
                )
            elif label == "formula_number":
                formula_number_index += 1
                matched_idx, matched_block = _match_best_ocr(
                    quad, ocr_blocks, used_ocr, threshold=0.20
                )
                if matched_idx is not None:
                    used_ocr.add(matched_idx)
                regions.append(
                    _region(
                        idx=formula_number_index,
                        kind="formula_number",
                        box=quad,
                        text=str((matched_block or {}).get("text") or ""),
                        latex="",
                        score=float(box.get("score") or 0.0),
                        source="formula_layout",
                    )
                )

        text_index = 0
        for idx, block in enumerate(ocr_blocks):
            if idx in used_ocr:
                continue
            quad = _to_quad(block.get("box") or [])
            if not quad:
                continue
            if any(
                item["type"] in {"formula", "formula_number"}
                and _overlap_ratio(quad, item["box"]) > 0.30
                for item in regions
            ):
                continue
            text_index += 1
            regions.append(
                _region(
                    idx=text_index,
                    kind="text",
                    box=quad,
                    text=str(block.get("text") or ""),
                    latex="",
                    score=float(block.get("score") or 0.0),
                    source="ocr_v6",
                )
            )

    regions.sort(key=_region_sort_key)
    counts = Counter(region["type"] for region in regions)
    return {
        "source": "formula_recognition",
        "mode": mode,
        "model": model_name,
        "regions": regions,
        "counts": dict(counts),
    }


def regions_to_ocr_blocks(regions: Optional[List[dict]]) -> List[Dict[str, Any]]:
    """Convert structured formula regions into host-compatible textBlocks."""
    source_regions = list(regions or [])
    blocks: List[Dict[str, Any]] = []
    for index, region in enumerate(source_regions):
        quad = _to_quad(region.get("box") or [])
        if not quad:
            continue
        text = str(region.get("text") or region.get("latex") or "")
        if not text:
            continue
        rect = _bbox(quad)
        height = max(1, rect[3] - rect[1])
        end = ""
        if index < len(source_regions) - 1:
            next_quad = _to_quad((source_regions[index + 1] or {}).get("box") or [])
            if next_quad:
                next_rect = _bbox(next_quad)
                end = " " if next_rect[1] <= rect[1] + int(height * 0.4) else "\n"
            else:
                end = "\n"
        blocks.append(
            {
                "box": quad,
                "text": text,
                "score": float(region.get("score") or 0.0),
                "end": end,
                "from": region.get("type") or "formula",
            }
        )
    return blocks
