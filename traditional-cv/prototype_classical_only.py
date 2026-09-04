"""
Prototype: pure classical CV isolated-cell contour extraction, NO model
(no YOLO box detection at all). Single whole-image findContours pass,
filtered by area + circularity + WBC-color -- no per-box Python loop.

Contours are returned EXACTLY as cv2.findContours produces them
(CHAIN_APPROX_NONE, every boundary pixel) -- no smoothing, no approxPolyDP
simplification.

Goal: match the semantics of the existing YOLO+CV pipeline (contour ONLY
for genuinely isolated single RBCs; touching/overlapping clusters and WBCs
get nothing) but without a model, targeting <30ms/tile.
"""
import sys
import time

import cv2
import numpy as np

RBC_HUE_MIN, RBC_HUE_MAX_WRAP = 130, 15
WBC_HUE_MIN, WBC_HUE_MAX = 120, 155
WBC_SAT_MIN = 40
WBC_REJECT_FRAC = 0.35
_CLOSE_KERNEL = np.ones((3, 3), np.uint8)


def _foreground_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, fg = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hue = hsv[:, :, 0]
    rbc_hue = (hue >= RBC_HUE_MIN) | (hue <= RBC_HUE_MAX_WRAP)
    fg[~rbc_hue] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return fg, hsv


def classical_isolated_cell_contours(img_bgr, min_area, max_area, min_circularity, debug=False):
    H, W = img_bgr.shape[:2]
    fg, hsv = _foreground_mask(img_bgr)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = []
    debug_rows = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        accept_geom = (min_area <= area <= max_area) and (circularity >= min_circularity)
        if debug:
            debug_rows.append((area, circularity))
        if not accept_geom:
            continue

        # WBC-color rejection: sample hue/sat inside this contour's own mask,
        # using a LOCAL bounding-rect crop (not a full-image-sized mask) --
        # allocating a full H*W array per candidate blob was the actual
        # bottleneck, not findContours itself
        bx, by, bw, bh = cv2.boundingRect(c)
        local_mask = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(local_mask, [c - [bx, by]], -1, 255, -1)
        region = local_mask > 0
        hsv_crop = hsv[by:by + bh, bx:bx + bw]
        hue = hsv_crop[:, :, 0][region]
        sat = hsv_crop[:, :, 1][region]
        if hue.size == 0:
            continue
        wbc_frac = ((hue >= WBC_HUE_MIN) & (hue <= WBC_HUE_MAX) & (sat >= WBC_SAT_MIN)).mean()
        if wbc_frac >= WBC_REJECT_FRAC:
            continue

        # no smoothing, no approxPolyDP -- the exact raw findContours
        # boundary (CHAIN_APPROX_NONE, every pixel), untouched
        pts = c.reshape(-1, 2).astype(np.float32)
        out.append(pts)

    if debug:
        return out, debug_rows
    return out


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Livo\Data - 2026\Rbc\others\7\Img_7_13.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("could not read", img_path)
        sys.exit(1)

    # first pass: gather area/circularity distribution to pick thresholds
    _, rows = classical_isolated_cell_contours(img, min_area=0, max_area=1e9, min_circularity=0, debug=True)
    areas = np.array([r[0] for r in rows])
    circ = np.array([r[1] for r in rows])
    print(f"{len(rows)} raw blobs found")
    print("area percentiles (5/25/50/75/95):", np.percentile(areas, [5, 25, 50, 75, 95]))
    print("circularity percentiles (5/25/50/75/95):", np.percentile(circ, [5, 25, 50, 75, 95]))
