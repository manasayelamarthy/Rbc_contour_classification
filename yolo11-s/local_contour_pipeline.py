"""
Redesigned box-to-contour step: fully local, per-box segmentation instead of
one whole-image Otsu + connected-components pass. Kills the bug where a WBC
(or any far-away region) could bleed into a nearby RBC's traced shape via a
shared global blob, and is more robust to per-region staining variation
since each box's threshold only ever has to explain its own small crop.

Per box:
  1. Crop locally (box + CROP_PAD, same padding as the old pipeline).
  2. Local Otsu on just that crop (+ the same RBC-hue restriction as before).
  3. Pick the connected component closest to the box's own center.
  4. Sanity-check it against the box's own size (not empty, not absurdly
     larger than the box) -- reject if implausible rather than trust a bad
     local threshold.
  5. Compare the resulting shape's actual pixels against nearby boxes' own
     shapes (not bounding-box overlap) to decide touching vs isolated.
  6. Smooth + simplify exactly as before.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as old_pipeline

RBC_HUE_MIN = old_pipeline.RBC_HUE_MIN
RBC_HUE_MAX_WRAP = old_pipeline.RBC_HUE_MAX_WRAP
CROP_PAD = old_pipeline.CROP_PAD
SMOOTH_WINDOW = old_pipeline.SMOOTH_WINDOW
SIMPLIFY_EPS = old_pipeline.SIMPLIFY_EPS
_CLOSE_KERNEL = np.ones((3, 3), np.uint8)

MIN_AREA_FRAC = 0.15     # reject if segmented shape is under 15% of the box's own area
MAX_SIZE_RATIO = 2.5     # reject if segmented shape is over 2.5x the box's own width/height
NEIGHBOR_MARGIN = 40     # px -- only compare against boxes whose (expanded) rect is within this


def _local_foreground_mask(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, fg = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hue = hsv[:, :, 0]
    rbc_hue = (hue >= RBC_HUE_MIN) | (hue <= RBC_HUE_MAX_WRAP)
    fg[~rbc_hue] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return fg


def _extract_local_shape(img_bgr, box):
    """Returns (mask_uint8_or_None, offset_x, offset_y) -- mask is in crop-local
    coords, offset converts it back to full-image coords."""
    x1, y1, x2, y2 = box
    H, W = img_bgr.shape[:2]
    cx1, cy1 = max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD)
    cx2, cy2 = min(W, x2 + CROP_PAD), min(H, y2 + CROP_PAD)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, cx1, cy1

    fg = _local_foreground_mask(crop)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels <= 1:
        return None, cx1, cy1

    box_cx, box_cy = (x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1
    best_lbl, best_dist = None, None
    for lbl in range(1, n_labels):
        cx, cy = centroids[lbl]
        d = (cx - box_cx) ** 2 + (cy - box_cy) ** 2
        if best_dist is None or d < best_dist:
            best_dist, best_lbl = d, lbl
    if best_lbl is None:
        return None, cx1, cy1

    box_w, box_h = x2 - x1, y2 - y1
    x, y, w, h, area = stats[best_lbl]
    if area < MIN_AREA_FRAC * box_w * box_h:
        return None, cx1, cy1
    if w > MAX_SIZE_RATIO * box_w or h > MAX_SIZE_RATIO * box_h:
        return None, cx1, cy1

    mask = (labels == best_lbl).astype(np.uint8) * 255
    return mask, cx1, cy1


def _shapes_touch(mask_a, off_a, mask_b, off_b):
    """True if the two LOCAL masks (each with its own image-space offset)
    share, or come within 1px of sharing, any pixel in global coordinates."""
    ax1, ay1 = off_a
    ax2, ay2 = ax1 + mask_a.shape[1], ay1 + mask_a.shape[0]
    bx1, by1 = off_b
    bx2, by2 = bx1 + mask_b.shape[1], by1 + mask_b.shape[0]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 - ix1 <= -2 or iy2 - iy1 <= -2:   # not even close
        return False

    # shared canvas covering the union, with 1px margin so "adjacent" counts as touching
    ux1, uy1 = min(ax1, bx1) - 1, min(ay1, by1) - 1
    ux2, uy2 = max(ax2, bx2) + 1, max(ay2, by2) + 1
    W, H = ux2 - ux1, uy2 - uy1
    if W <= 0 or H <= 0 or W > 4000 or H > 4000:
        return False

    canvas_a = np.zeros((H, W), dtype=np.uint8)
    canvas_a[ay1 - uy1: ay1 - uy1 + mask_a.shape[0], ax1 - ux1: ax1 - ux1 + mask_a.shape[1]] = mask_a
    canvas_a = cv2.dilate(canvas_a, np.ones((3, 3), np.uint8), iterations=1)

    canvas_b = np.zeros((H, W), dtype=np.uint8)
    canvas_b[by1 - uy1: by1 - uy1 + mask_b.shape[0], bx1 - ux1: bx1 - ux1 + mask_b.shape[1]] = mask_b

    return bool(np.any((canvas_a > 0) & (canvas_b > 0)))


def isolated_cell_contours_local(img_bgr, boxes, confidences):
    keep = old_pipeline._dedup_boxes(boxes, confidences)
    kept_idx = [i for i, k in enumerate(keep) if k]

    shapes = {}   # idx -> (mask, off_x, off_y) or None
    for i in kept_idx:
        shapes[i] = _extract_local_shape(img_bgr, boxes[i])

    # cheap spatial pre-filter: only compare boxes whose expanded rects overlap
    def expanded(b):
        x1, y1, x2, y2 = b
        return (x1 - NEIGHBOR_MARGIN, y1 - NEIGHBOR_MARGIN, x2 + NEIGHBOR_MARGIN, y2 + NEIGHBOR_MARGIN)

    touching = set()
    for a_pos, i in enumerate(kept_idx):
        if shapes[i][0] is None:
            continue
        ex1, ey1, ex2, ey2 = expanded(boxes[i])
        for j in kept_idx[a_pos + 1:]:
            if shapes[j][0] is None:
                continue
            fx1, fy1, fx2, fy2 = boxes[j]
            if fx2 < ex1 or fx1 > ex2 or fy2 < ey1 or fy1 > ey2:
                continue
            mask_a, ox_a, oy_a = shapes[i]
            mask_b, ox_b, oy_b = shapes[j]
            if _shapes_touch(mask_a, (ox_a, oy_a), mask_b, (ox_b, oy_b)):
                touching.add(i)
                touching.add(j)

    out = [None] * len(boxes)
    for i in kept_idx:
        if i in touching:
            continue
        mask, ox, oy = shapes[i]
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        best = max(contours, key=cv2.contourArea)
        pts = best.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += ox
        pts[:, 1] += oy
        smoothed = old_pipeline._smooth_closed_contour(pts)
        simplified = cv2.approxPolyDP(smoothed.reshape(-1, 1, 2), SIMPLIFY_EPS, True)
        out[i] = simplified.reshape(-1, 2)
    return out
