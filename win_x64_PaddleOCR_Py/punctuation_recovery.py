"""Conservative image-evidenced full-stop recovery for vertical OCR blocks.

OCR text is never changed from linguistic context.  This module inserts only
the CJK full stop ``。`` when an actual small annular connected component is
found in an otherwise aligned vertical character cell.  The returned metadata
is deliberately JSON-safe so raw/preview/document traces retain the evidence.
"""
from __future__ import annotations

import statistics

import cv2
import numpy as np


SOURCE = "image_connected_component"


def _rect_from_box(box):
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _vertical_block(block):
    text = block.get("text") if isinstance(block, dict) else ""
    rect = _rect_from_box(block.get("box")) if isinstance(block, dict) else None
    if not isinstance(text, str) or len(text) < 4 or not rect:
        return None
    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    if width <= 2 or height < width * 2.4:
        return None
    return x0, y0, x1, y1, text


def _annular_candidates(binary):
    """Return small contours with a real enclosed white hole (not speckle)."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    result = []
    for index, contour in enumerate(contours):
        child = int(hierarchy[index][2])
        if child < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        child_area = float(cv2.contourArea(contours[child]))
        if area <= 0 or child_area <= 0:
            continue
        result.append((x, y, w, h, area, child_area))
    return result


def _ink_run_heights(binary):
    """Height of the foreground row-run containing each y coordinate."""
    rows = np.any(binary > 0, axis=1)
    heights = [0] * len(rows)
    start = None
    for index, has_ink in enumerate(rows.tolist() + [False]):
        if has_ink and start is None:
            start = index
        elif not has_ink and start is not None:
            for row in range(start, index):
                heights[row] = index - start
            start = None
    return heights


def recover_vertical_full_stops(image_bgr, blocks, *, enabled=True):
    """Mutate and return OCR blocks only when physical ``。`` evidence is strong.

    Requirements intentionally stack: vertical geometry, annular connected
    component, punctuation-scale dimensions, grid-cell alignment, and a low
    ink punctuation cell compared with neighbouring character cells.  Any
    uncertain case is returned unchanged.
    """
    if not enabled or image_bgr is None or not isinstance(blocks, list):
        return blocks
    # Avoid even a full-page grayscale conversion for the overwhelmingly common
    # horizontal OCR request.
    if not any(_vertical_block(block) for block in blocks):
        return blocks
    height_img, width_img = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    for block in blocks:
        geometry = _vertical_block(block)
        if not geometry or block.get("punctuation_recovery"):
            continue
        x0, y0, x1, y1, text = geometry
        ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        ix1, iy1 = min(width_img, int(np.ceil(x1))), min(height_img, int(np.ceil(y1)))
        if ix1 - ix0 < 6 or iy1 - iy0 < 12:
            continue
        roi = gray[iy0:iy1, ix0:ix1]
        # Otsu is resilient to the scanned page's local paper colour.  The
        # Do not morphology-open here: at 150dpi it can break the thin annulus
        # that is the very evidence we need to preserve.
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        visual_cells = len(text)
        cell = (iy1 - iy0) / visual_cells
        if cell < 7:
            continue
        ink_per_cell = [
            int(np.count_nonzero(binary[max(0, round(i * cell)): min(binary.shape[0], round((i + 1) * cell)), :]))
            for i in range(visual_cells)
        ]
        median_ink = statistics.median(ink_per_cell)
        if median_ink < 8:
            continue
        run_heights = _ink_run_heights(binary)
        normal_run_heights = [height for height in run_heights if height]
        median_run_height = statistics.median(normal_run_heights)
        accepted_candidates = []
        for cx, cy, w, h, outer_area, hole_area in _annular_candidates(binary):
            ratio = hole_area / outer_area
            # Full stop is compact, close to square, and has a visible hole.
            if not (0.18 <= w / cell <= 0.72 and 0.18 <= h / cell <= 0.72):
                continue
            if not (0.55 <= w / max(h, 1) <= 1.65 and 0.10 <= ratio <= 0.70):
                continue
            center = cy + h / 2
            # OCR's box spans the recognised glyph sequence while the omitted
            # punctuation consumes an unrecognised visual cell.  Map the
            # isolated component by its normalised vertical rank (rather than
            # blindly assuming an extra cell at either boundary).
            slot = int(round(center / cell))
            if slot <= 0 or slot >= len(text):
                continue  # never fabricate at a block boundary
            expected_center = slot * cell
            if abs(center - expected_center) > cell * 0.32:
                continue
            center_row = min(len(run_heights) - 1, max(0, int(round(center))))
            if run_heights[center_row] > median_run_height * 0.55:
                continue  # a hole inside a normal-sized Han glyph
            # A Chinese character may contain a small hole; its complete cell
            # is ink-heavy.  A stand-alone full stop cell is comparatively low.
            if ink_per_cell[slot] > median_ink * 0.72:
                continue
            if text[slot - 1: slot + 1].find("。") >= 0:
                continue
            confidence = round(min(0.99, 0.80 + 0.10 * (1 - abs(center - expected_center) / (cell * 0.28)) + 0.10 * min(1.0, ratio / 0.35)), 3)
            accepted_candidates.append((slot, cx, cy, w, h, confidence))
        # Multiple plausible rings mean the image evidence is ambiguous.  Do
        # not depend on OpenCV contour order and never guess which ring is "。".
        if len(accepted_candidates) != 1:
            continue
        slot, cx, cy, w, h, confidence = accepted_candidates[0]
        component_box = [
            [ix0 + cx, iy0 + cy], [ix0 + cx + w, iy0 + cy],
            [ix0 + cx + w, iy0 + cy + h], [ix0 + cx, iy0 + cy + h],
        ]
        block["text"] = text[:slot] + "。" + text[slot:]
        block["punctuation_recovery"] = [{
            "character": "。", "source": SOURCE, "confidence": confidence,
            "bbox": component_box, "insert_index": slot,
        }]
    return blocks
