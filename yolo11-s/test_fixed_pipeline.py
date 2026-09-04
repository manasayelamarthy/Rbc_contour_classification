import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as pipeline

SOURCE = Path(r"F:\Livo\Data - 2026\Rbc\7")
TEST_IMAGES = ["Img_0_24", "Img_74_24", "Img_74_23", "Img_0_23", "Img_1_24", "Img_50_24"]
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\fixed_pipeline_test")
OUT_DIR.mkdir(exist_ok=True)

model = pipeline.YOLO(str(pipeline.BEST_PT))
_warm = cv2.imread(str(SOURCE / "Img_0_23.jpg"))
model.predict(_warm, conf=pipeline.CONF_THRES, iou=pipeline.IOU_THRES, imgsz=pipeline.IMGSZ, verbose=False)

print(f"{'image':<12s} {'boxes':>6s} {'contours':>9s} {'contour_ms':>11s} {'total_ms':>9s}")
for stem in TEST_IMAGES:
    img_path = SOURCE / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    dets, timing = pipeline.detect_and_contour(model, img)
    n_contours = sum(1 for d in dets if "contour" in d)
    print(f"{stem:<12s} {len(dets):6d} {n_contours:9d} {timing['contour_ms']:11.1f} {timing['total_ms']:9.1f}")

    vis = pipeline.draw_contours(img, dets)
    cv2.imwrite(str(OUT_DIR / f"{stem}_FIXED.jpg"), vis)

print(f"\nSaved -> {OUT_DIR}")
