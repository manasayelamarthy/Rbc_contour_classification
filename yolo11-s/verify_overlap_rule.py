import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as pipeline

SOURCE = Path(r"F:\Livo\Data - 2026\Rbc\7")
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\fixed_pipeline_test")
OUT_DIR.mkdir(exist_ok=True)

model = pipeline.YOLO(str(pipeline.BEST_PT))


def bbox_xyxy(d):
    b = d["bbox"]
    return b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]


stem = "Img_0_24"
img = cv2.imread(str(SOURCE / f"{stem}.jpg"))
dets, timing = pipeline.detect_and_contour(model, img)

no_contour = [d for d in dets if "contour" not in d]
with_contour = [d for d in dets if "contour" in d]
print(f"{stem}: {len(dets)} boxes, {len(with_contour)} with contour, {len(no_contour)} without")

# pairs among no-contour boxes whose boxes actually overlap (candidates for
# the new cell-overlap exclusion rule)
pairs = []
for i in range(len(no_contour)):
    x1a, y1a, x2a, y2a = bbox_xyxy(no_contour[i])
    for j in range(i + 1, len(no_contour)):
        x1b, y1b, x2b, y2b = bbox_xyxy(no_contour[j])
        ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
        ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        if iw * ih > 0:
            pairs.append((i, j, iw * ih))

pairs.sort(key=lambda p: -p[2])
print(f"found {len(pairs)} overlapping-box no-contour pairs")


def save_pair_crop(idx, i, j, items, path):
    xa1, ya1, xa2, ya2 = bbox_xyxy(items[i])
    xb1, yb1, xb2, yb2 = bbox_xyxy(items[j])
    cx1 = max(0, int(min(xa1, xb1)) - 20)
    cy1 = max(0, int(min(ya1, yb1)) - 20)
    cx2 = int(max(xa2, xb2)) + 20
    cy2 = int(max(ya2, yb2)) + 20
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (int(xa1 - cx1), int(ya1 - cy1)), (int(xa2 - cx1), int(ya2 - cy1)), (0, 0, 255), 1)
    cv2.rectangle(crop, (int(xb1 - cx1), int(yb1 - cy1)), (int(xb2 - cx1), int(yb2 - cy1)), (0, 255, 255), 1)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), crop)
    print(f"saved {path}")


for k, (i, j, area) in enumerate(pairs[:4]):
    save_pair_crop(k, i, j, no_contour, OUT_DIR / f"cellmask_excluded_{k}.jpg")

# also find pairs among WITH-contour boxes whose boxes overlap (these should
# now correctly get contours if their box-overlap is just padding, not real
# cell overlap)
wc_pairs = []
for i in range(len(with_contour)):
    x1a, y1a, x2a, y2a = bbox_xyxy(with_contour[i])
    for j in range(i + 1, len(with_contour)):
        x1b, y1b, x2b, y2b = bbox_xyxy(with_contour[j])
        ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
        ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        if iw * ih > 0:
            wc_pairs.append((i, j, iw * ih))

wc_pairs.sort(key=lambda p: -p[2])
print(f"found {len(wc_pairs)} overlapping-box WITH-contour pairs (both boxes still got a contour)")
for k, (i, j, area) in enumerate(wc_pairs[:3]):
    save_pair_crop(k, i, j, with_contour, OUT_DIR / f"cellmask_kept_despite_box_overlap_{k}.jpg")
