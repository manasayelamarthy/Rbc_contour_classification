import time
from pathlib import Path

import cv2
import numpy as np

import local_contour_pipeline as new_pipeline
import sys
sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as old_pipeline

SOURCE = Path(r"F:\Livo\Data - 2026\Rbc\7")
TEST_IMAGES = ["Img_0_24", "Img_74_24", "Img_74_23", "Img_0_23", "Img_1_24", "Img_50_24"]
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\local_pipeline_test")
OUT_DIR.mkdir(exist_ok=True)

model = old_pipeline.YOLO(str(old_pipeline.BEST_PT))

# warm up
_warm = cv2.imread(str(SOURCE / "Img_0_23.jpg"))
model.predict(_warm, conf=old_pipeline.CONF_THRES, iou=old_pipeline.IOU_THRES, imgsz=old_pipeline.IMGSZ, verbose=False)

print(f"{'image':<12s} {'boxes':>6s} {'old_n':>6s} {'new_n':>6s} {'old_ms':>8s} {'new_ms':>8s}")
for stem in TEST_IMAGES:
    img_path = SOURCE / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  SKIP (missing): {stem}")
        continue
    H, W = img.shape[:2]

    results = model.predict(img, conf=old_pipeline.CONF_THRES, iou=old_pipeline.IOU_THRES, imgsz=old_pipeline.IMGSZ, verbose=False)
    boxes, confidences = [], []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            boxes.append((x1, y1, x2, y2))
            confidences.append(round(float(box.conf[0]), 4))

    REPEATS = 3
    old_times, new_times = [], []
    for _ in range(REPEATS):
        t = time.perf_counter()
        old_contours = old_pipeline.isolated_cell_contours(img, boxes, confidences)
        old_times.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        new_contours = new_pipeline.isolated_cell_contours_local(img, boxes, confidences)
        new_times.append((time.perf_counter() - t) * 1000)

    old_med = sorted(old_times)[REPEATS // 2]
    new_med = sorted(new_times)[REPEATS // 2]
    n_old = sum(1 for c in old_contours if c is not None)
    n_new = sum(1 for c in new_contours if c is not None)
    print(f"{stem:<12s} {len(boxes):6d} {n_old:6d} {n_new:6d} {old_med:8.1f} {new_med:8.1f}")

    vis_old = img.copy()
    for c in old_contours:
        if c is not None:
            cv2.polylines(vis_old, [c.astype(np.int32)], True, (0, 160, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / f"{stem}_OLD.jpg"), vis_old)

    vis_new = img.copy()
    for c in new_contours:
        if c is not None:
            cv2.polylines(vis_new, [c.astype(np.int32)], True, (0, 160, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / f"{stem}_NEW.jpg"), vis_new)

print(f"\nSaved comparisons -> {OUT_DIR}")
