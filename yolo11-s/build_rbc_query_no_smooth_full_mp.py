"""
Generates ONLY the no-smooth parquet deliverable for the full monolayer image
set, using the raw cv2.findContours output (no moving-average smoothing, no
approxPolyDP simplification -- ENABLE_SMOOTHING=False). Reuses the shared
encoding/schema/manifest helpers from build_rbc_query_full_smooth_compare_mp.py.
"""
import csv
import json
import multiprocessing
import time

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import build_rbc_query_full_smooth_compare_mp as base

pipeline = base.pipeline
SOURCE_DIR = base.SOURCE_DIR
IMAGE_LIST_CSV = base.IMAGE_LIST_CSV
OUT_DIR = base.OUT_DIR
N_WORKERS = base.N_WORKERS
EDGE_MARGIN = base.EDGE_MARGIN
CLASS_BIT_EDGE = base.CLASS_BIT_EDGE
_build_cell_record = base._build_cell_record
write_output = base.write_output

_worker_model = None


def _init_worker(_unused=None):
    global _worker_model
    _worker_model = pipeline.YOLO(str(pipeline.BEST_PT))
    pipeline.ENABLE_SMOOTHING = False


def _process_image(task):
    idx, col, row, img_path_str = task
    img = cv2.imread(img_path_str)
    if img is None:
        return idx, col, row, None, None, []
    H, W = img.shape[:2]
    tile_off_x, tile_off_y = col * W, row * H

    results = _worker_model.predict(img, conf=pipeline.CONF_THRES, iou=pipeline.IOU_THRES,
                                     imgsz=pipeline.IMGSZ, verbose=False)
    boxes, confidences = [], []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            boxes.append((x1, y1, x2, y2))
            confidences.append(round(float(box.conf[0]), 4))

    contours = pipeline.isolated_cell_contours(img, boxes, confidences)

    candidates = []
    for i, pts in enumerate(contours):
        if pts is None:
            continue
        bx1, by1, bx2, by2 = boxes[i]
        candidates.append((by1, bx1, pts))
    candidates.sort(key=lambda c: (c[0], c[1]))   # stable: box top-left corner

    cell_records = []
    for (_by1, _bx1, pts) in candidates:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 3:
            continue
        min_x = int(round(pts[:, 0].min()))
        min_y = int(round(pts[:, 1].min()))
        max_x = int(round(pts[:, 0].max()))
        max_y = int(round(pts[:, 1].max()))
        edge = (
            min_x <= EDGE_MARGIN or min_y <= EDGE_MARGIN
            or max_x >= W - EDGE_MARGIN or max_y >= H - EDGE_MARGIN
        )
        class_code = CLASS_BIT_EDGE if edge else 0
        record, n = _build_cell_record(pts, tile_off_x, tile_off_y, class_code)
        cell_records.append((record, n))

    return idx, col, row, W, H, cell_records


def run_pass(pool, img_paths):
    tasks = [(i, col, row, str(p)) for i, (col, row, p) in enumerate(img_paths)]
    results = [None] * len(img_paths)

    t_start = time.perf_counter()
    done = 0
    for idx, col, row, W, H, cell_records in pool.imap_unordered(_process_image, tasks, chunksize=8):
        results[idx] = cell_records
        done += 1
        if done % 500 == 0 or done == len(tasks):
            elapsed_so_far = time.perf_counter() - t_start
            print(f"  {done:5d}/{len(tasks)} images done  elapsed={elapsed_so_far:7.1f}s")

    rows = []
    point_counts = []
    next_id = 0
    total_points = 0
    for cell_records in results:
        if cell_records is None:
            continue
        for record, n in cell_records:
            record["rbc_id"] = next_id
            rows.append(record)
            point_counts.append(n)
            total_points += n
            next_id += 1

    elapsed = time.perf_counter() - t_start
    print(f"{len(rows)} cells, {total_points} total exact-contour points, {elapsed:.1f}s")

    size = write_output(rows, point_counts, total_points,
                         OUT_DIR / "rbc_query_no_smooth.parquet",
                         OUT_DIR / "rbc_query_no_smooth_manifest.json", "no-smooth")
    return elapsed, size, len(rows), total_points


def main():
    img_paths = []
    with open(IMAGE_LIST_CSV, newline="") as f:
        for entry in csv.DictReader(f):
            col, row = int(entry["col"]), int(entry["row"])
            p = SOURCE_DIR / f"Img_{col}_{row}.jpg"
            if p.exists():
                img_paths.append((col, row, p))
    img_paths.sort(key=lambda t: (t[0], t[1]))
    print(f"{len(img_paths)} images from {IMAGE_LIST_CSV.name}  ({N_WORKERS} worker processes, no-smooth only)")

    with multiprocessing.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        elapsed, size, n_rows, n_points = run_pass(pool, img_paths)

    print("\n=== SUMMARY ===")
    print(f"no_smooth  time={elapsed:.1f}s  size={size:,} bytes  rows={n_rows}  points={n_points}")


if __name__ == "__main__":
    main()
