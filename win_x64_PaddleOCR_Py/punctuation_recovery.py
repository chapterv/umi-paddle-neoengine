"""Conservative image-evidenced punctuation recovery for vertical OCR blocks.

OCR text is never changed from linguistic context.  This module inserts only
``。``, ``，``, and ``：`` when their physical connected-component shape is
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


def _slot_from_word_boxes(block, candidate_y, fallback):
    """Map physical punctuation to the OCR text using recognition word boxes."""
    text = block.get("text")
    word_texts = block.get("_word_texts")
    word_boxes = block.get("_word_boxes")
    if (
        not isinstance(text, str)
        or not isinstance(word_texts, list)
        or not isinstance(word_boxes, list)
        or len(word_texts) != len(word_boxes)
        or "".join(str(word) for word in word_texts) != text
    ):
        return fallback
    slot = 0
    for word, box in zip(word_texts, word_boxes):
        rect = _rect_from_box(box)
        if not rect:
            return fallback
        _, y0, _, y1 = rect
        if (y0 + y1) / 2 >= candidate_y:
            break
        slot += len(str(word))
    return slot


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


def _connected_components(binary):
    """Return isolated foreground components without relying on contour order."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [
        tuple(int(value) for value in stats[index])
        for index in range(1, count)
    ]


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


def recover_vertical_punctuation(image_bgr, blocks, *, enabled=True):
    """Mutate OCR blocks only when physical punctuation evidence is strong.

    Requirements intentionally stack: vertical geometry, punctuation-specific
    connected-component shape, grid-cell alignment, low ink compared with
    neighbouring character cells, and recognition word-box alignment when the
    OCR backend provides it.  Any uncertain case is returned unchanged.
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
            insert_slot = _slot_from_word_boxes(
                block, iy0 + center, slot
            )
            if text[max(0, insert_slot - 1): insert_slot + 1].find("。") >= 0:
                continue
            confidence = round(min(0.99, 0.80 + 0.10 * (1 - abs(center - expected_center) / (cell * 0.28)) + 0.10 * min(1.0, ratio / 0.35)), 3)
            accepted_candidates.append(
                (insert_slot, cx, cy, w, h, confidence)
            )
        # Multiple plausible rings mean the image evidence is ambiguous.  Do
        # not depend on OpenCV contour order and never guess which ring is "。".
        if len(accepted_candidates) > 1:
            continue
        character = "。"
        if accepted_candidates:
            slot, cx, cy, w, h, confidence = accepted_candidates[0]
        else:
            comma_candidates = []
            components = _connected_components(binary)
            for cx, cy, w, h, area in components:
                center_x = cx + w / 2
                center_y = cy + h / 2
                slot = int(center_y // cell)
                if slot <= 0 or slot >= len(text):
                    continue
                if not (0.12 <= w / cell <= 0.35 and 0.25 <= h / cell <= 0.55):
                    continue
                if not (1.45 <= h / max(w, 1) <= 3.5):
                    continue
                if not (0.35 <= center_x / binary.shape[1] <= 0.65):
                    continue
                if not (0.025 <= area / (cell * cell) <= 0.10):
                    continue
                expected_center = slot * cell
                if abs(center_y - expected_center) > cell * 0.36:
                    continue
                center_row = min(
                    len(run_heights) - 1, max(0, int(round(center_y)))
                )
                if run_heights[center_row] > median_run_height * 0.55:
                    continue
                if ink_per_cell[slot] > median_ink * 0.50:
                    continue
                insert_slot = _slot_from_word_boxes(
                    block, iy0 + center_y, slot
                )
                if text[
                    max(0, insert_slot - 1): insert_slot + 1
                ].find("，") >= 0:
                    continue
                confidence = round(
                    min(
                        0.97,
                        0.80
                        + 0.09
                        * (1 - abs(center_y - expected_center) / (cell * 0.36))
                        + 0.08 * (1 - ink_per_cell[slot] / (median_ink * 0.50)),
                    ),
                    3,
                )
                comma_candidates.append(
                    (insert_slot, cx, cy, w, h, confidence)
                )
            if len(comma_candidates) > 1:
                continue
            if comma_candidates:
                character = "，"
                slot, cx, cy, w, h, confidence = comma_candidates[0]
            else:
                colon_candidates = []
                for upper_index, upper in enumerate(components):
                    ux, uy, uw, uh, upper_area = upper
                    upper_center_x = ux + uw / 2
                    upper_center_y = uy + uh / 2
                    if not (
                        0.08 <= uw / cell <= 0.22
                        and 0.10 <= uh / cell <= 0.25
                    ):
                        continue
                    if not (0.55 <= uw / max(uh, 1) <= 1.50):
                        continue
                    if not (0.010 <= upper_area / (cell * cell) <= 0.050):
                        continue
                    for lower in components[upper_index + 1:]:
                        lx, ly, lw, lh, lower_area = lower
                        lower_center_x = lx + lw / 2
                        lower_center_y = ly + lh / 2
                        if lower_center_y <= upper_center_y:
                            continue
                        if not (
                            0.08 <= lw / cell <= 0.22
                            and 0.10 <= lh / cell <= 0.25
                        ):
                            continue
                        if not (0.55 <= lw / max(lh, 1) <= 1.50):
                            continue
                        if not (
                            0.010 <= lower_area / (cell * cell) <= 0.050
                        ):
                            continue
                        if abs(upper_center_x - lower_center_x) > cell * 0.08:
                            continue
                        gap = ly - (uy + uh)
                        if not (0.04 <= gap / cell <= 0.22):
                            continue
                        center_x = (upper_center_x + lower_center_x) / 2
                        center_y = (upper_center_y + lower_center_y) / 2
                        if not (
                            0.35 <= center_x / binary.shape[1] <= 0.65
                        ):
                            continue
                        slot = int(center_y // cell)
                        if slot <= 0 or slot >= len(text):
                            continue
                        if ink_per_cell[slot] > median_ink * 0.55:
                            continue
                        upper_row = min(
                            len(run_heights) - 1,
                            max(0, int(round(upper_center_y))),
                        )
                        lower_row = min(
                            len(run_heights) - 1,
                            max(0, int(round(lower_center_y))),
                        )
                        if (
                            run_heights[upper_row] > median_run_height * 0.35
                            or run_heights[lower_row]
                            > median_run_height * 0.35
                        ):
                            continue
                        insert_slot = _slot_from_word_boxes(
                            block, iy0 + center_y, slot
                        )
                        if text[
                            max(0, insert_slot - 1): insert_slot + 1
                        ].find("：") >= 0:
                            continue
                        cx = min(ux, lx)
                        cy = min(uy, ly)
                        right = max(ux + uw, lx + lw)
                        bottom = max(uy + uh, ly + lh)
                        w, h = right - cx, bottom - cy
                        confidence = round(
                            min(
                                0.98,
                                0.84
                                + 0.07
                                * (
                                    1
                                    - abs(
                                        upper_center_x - lower_center_x
                                    )
                                    / (cell * 0.08)
                                )
                                + 0.07
                                * (
                                    1
                                    - ink_per_cell[slot]
                                    / (median_ink * 0.55)
                                ),
                            ),
                            3,
                        )
                        colon_candidates.append(
                            (insert_slot, cx, cy, w, h, confidence)
                        )
                if len(colon_candidates) != 1:
                    continue
                character = "："
                slot, cx, cy, w, h, confidence = colon_candidates[0]
        component_box = [
            [ix0 + cx, iy0 + cy], [ix0 + cx + w, iy0 + cy],
            [ix0 + cx + w, iy0 + cy + h], [ix0 + cx, iy0 + cy + h],
        ]
        block["text"] = text[:slot] + character + text[slot:]
        block["punctuation_recovery"] = [{
            "character": character, "source": SOURCE, "confidence": confidence,
            "bbox": component_box, "insert_index": slot,
        }]
    return blocks


def recover_vertical_full_stops(image_bgr, blocks, *, enabled=True):
    """Backward-compatible entry point retained for existing host patches."""
    return recover_vertical_punctuation(image_bgr, blocks, enabled=enabled)
