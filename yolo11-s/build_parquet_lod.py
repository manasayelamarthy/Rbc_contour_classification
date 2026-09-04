
import heapq
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as pipeline   # box detector + classical-CV
                                                     # contour extraction -- the
                                                     # seg-model route this script
                                                     # originally used was dropped
                                                     # (5-6x slower, faceted output)

BEST_WEIGHTS = pipeline.BEST_PT   # F:/.../rbc_yolo/yolo11s_1024_singlecls_expanded-2/weights/best.pt
SOURCE_DIR = Path(r"F:\Livo\Data - 2026\Rbc\7")   # master tile pool, addressed Img_{y}_{x}.jpg
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\parquet_output")
OUT_FILE = OUT_DIR / "rbc_contours_lod.parquet"

TEST_X = [23, 24]     # test slice: these 2 tile columns, every row
TEST_Y = range(128)

LEVEL1_MIN, LEVEL1_MAX = 5, 6
LEVEL_BUDGETS = {4: 8, 5: 16, 6: 24}   # cumulative point counts
EPSILON_FRACTION = 0.05                 # fixed tolerance -- classification-based multiplier skipped


def importance_order(pts):
    """Rank every point index in a closed polygon by shape importance.
    Starts from the two farthest-apart points (the anchors), then
    repeatedly finds -- across ALL pending arcs, globally -- whichever
    single point has the largest perpendicular deviation from its arc's
    anchor-to-anchor line, adds it to the order, and splits that arc in two."""
    n = len(pts)
    if n <= 2:
        return list(range(n))

    best = (-1.0, 0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(*(pts[i] - pts[j]))
            if d > best[0]:
                best = (d, i, j)
    a, b = best[1], best[2]

    order = [a, b]
    seen = {a, b}

    def arc_indices(i, j):
        idx = []
        k = (i + 1) % n
        while k != j:
            idx.append(k)
            k = (k + 1) % n
        return idx

    def farthest_in_arc(arc, i, j):
        if not arc:
            return None
        p1, p2 = pts[i], pts[j]
        seg = p2 - p1
        seg_len = np.hypot(*seg)
        if seg_len == 0:
            dists = [np.hypot(*(pts[k] - p1)) for k in arc]
        else:
            dists = [abs(np.cross(seg, pts[k] - p1)) / seg_len for k in arc]
        m = int(np.argmax(dists))
        return dists[m], arc[m]

    heap = []
    counter = 0
    for (i, j) in [(a, b), (b, a)]:
        arc = arc_indices(i, j)
        res = farthest_in_arc(arc, i, j)
        if res:
            dist, pt = res
            counter += 1
            heapq.heappush(heap, (-dist, counter, i, j, arc, pt))

    while heap:
        _, _, i, j, arc, pt = heapq.heappop(heap)
        if pt in seen:
            continue
        order.append(pt)
        seen.add(pt)
        left = [k for k in arc_indices(i, pt) if k in arc]
        right = [k for k in arc_indices(pt, j) if k in arc]
        for (ii, jj, sub) in [(i, pt, left), (pt, j, right)]:
            res = farthest_in_arc(sub, ii, jj)
            if res:
                dist, p2 = res
                counter += 1
                heapq.heappush(heap, (-dist, counter, ii, jj, sub, p2))

    for k in range(n):
        if k not in seen:
            order.append(k)
    return order


def level1_count(pts):
    peri = cv2.arcLength(pts.astype(np.float32), True)
    eps = peri * EPSILON_FRACTION
    approx = cv2.approxPolyDP(pts.astype(np.float32), eps, True)
    n = len(pts)
    return min(n, max(LEVEL1_MIN, min(LEVEL1_MAX, len(approx))))


def points_payload(pts, indices):
    return [{"point_index": int(idx), "x": float(pts[idx][0]), "y": float(pts[idx][1])} for idx in indices]


def _zigzag(n):
    """Map signed ints to unsigned so small negative and positive deltas both
    encode to few bytes: 0,-1,1,-2,2,... -> 0,1,2,3,4,..."""
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _varint(n):
    """Base-128 varint: 7 payload bits per byte, top bit = 'more bytes follow'."""
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def encode_level7_path(pts):
    """Full, unsimplified contour (every point, in original model-output
    order) packed as a compact binary blob: [uint16 point count][per-point
    zigzag-varint-encoded (dx,dy) delta from the previous point]. This is
    the complete contour -- it duplicates every point already stored in
    levels 1/4/5/6, by design, per the reference format."""
    n = len(pts)
    out = bytearray()
    out += n.to_bytes(2, "little")
    prev_x, prev_y = 0, 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        dx, dy = xi - prev_x, yi - prev_y
        out += _varint(_zigzag(dx))
        out += _varint(_zigzag(dy))
        prev_x, prev_y = xi, yi
    return bytes(out)


def process_image(model, img_path):
    """Runs the box detector + classical-CV contour pipeline on one image and
    LOD-encodes every isolated cell's contour. Touching/clustered cells (no
    'contour' key from the pipeline) are skipped, same as everywhere else in
    this project -- only single, separate RBCs are meant to appear here."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [], {"detect_ms": 0.0, "contour_ms": 0.0, "total_ms": 0.0}
    dets, timing = pipeline.detect_and_contour(model, img)

    rows = []
    for rbc_id, det in enumerate(dets):
        if "contour" not in det:
            continue
        pts = np.array(det["contour"], dtype=np.float64)
        n = len(pts)
        if n < 3:
            continue
        order = importance_order(pts)
        l1n = level1_count(pts)
        c4 = min(n, LEVEL_BUDGETS[4])
        c5 = min(n, LEVEL_BUDGETS[5])
        c6 = min(n, LEVEL_BUDGETS[6])

        rows.append({
            "image": img_path.name,
            "rbc_id": rbc_id,
            "min_x": float(pts[:, 0].min()),
            "min_y": float(pts[:, 1].min()),
            "width": float(pts[:, 0].max() - pts[:, 0].min()),
            "height": float(pts[:, 1].max() - pts[:, 1].min()),
            "level1_points": points_payload(pts, order[:l1n]),
            "level4_additions": points_payload(pts, order[l1n:c4]),
            "level5_additions": points_payload(pts, order[c4:c5]),
            "level6_additions": points_payload(pts, order[c5:c6]),
            "level7_path": encode_level7_path(pts),
        })
    return rows, timing


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST_WEIGHTS))
    print(f"Model: {BEST_WEIGHTS}")

    img_paths = []
    for y in TEST_Y:
        for x in TEST_X:
            p = SOURCE_DIR / f"Img_{y}_{x}.jpg"
            if p.exists():
                img_paths.append(p)
    print(f"test set: {len(img_paths)} images from {SOURCE_DIR} (x in {TEST_X}, y 0-{max(TEST_Y)})")

    # warm up (excluded from the timing total below)
    if img_paths:
        _ = pipeline.detect_and_contour(model, cv2.imread(str(img_paths[0])))

    all_rows = []
    all_timings = []
    t_start = time.perf_counter()
    for idx, img_path in enumerate(img_paths, 1):
        rows, timing = process_image(model, img_path)
        all_rows.extend(rows)
        all_timings.append(timing)
        print(f"  [{idx:3d}/{len(img_paths)}] {img_path.name:<16s}  "
              f"cells={len(rows):4d}  total_rows={len(all_rows):5d}  "
              f"detect={timing['detect_ms']:6.1f}ms  contour={timing['contour_ms']:6.1f}ms")
    total_elapsed_s = time.perf_counter() - t_start

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_FILE, index=False)
    file_size = OUT_FILE.stat().st_size

    n = len(all_timings)
    avg_detect = sum(t["detect_ms"] for t in all_timings) / n if n else 0
    avg_contour = sum(t["contour_ms"] for t in all_timings) / n if n else 0

    print(f"\n{len(df)} rows ({len(img_paths)} images) -> {OUT_FILE}")
    print(f"Parquet file size: {file_size:,} bytes ({file_size/1024:.1f} KB / {file_size/1024/1024:.3f} MB)")
    print(f"Total wall time (incl. warmup-free inference for all images): {total_elapsed_s:.2f}s "
          f"({total_elapsed_s/len(img_paths)*1000:.1f}ms/image average)")
    print(f"Per-image average: detect={avg_detect:.1f}ms  contour={avg_contour:.1f}ms")


if __name__ == "__main__":
    main()
