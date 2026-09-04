"""
Multiprocess version of build_rbc_query_full_smooth_compare.py.

Builds TWO parquet deliverables (raw / smoothed) for the monolayer-region
image set listed in monolayer1_level7_fovs_detection_only.csv, in a SINGLE
pass per image: box detection runs once, then isolated_cell_contours is run
twice (smoothing off, smoothing on) against the SAME boxes so the two
contour lists stay index-aligned box-for-box. Cells are sorted by a stable
key derived from the ORIGINAL BOX position (identical regardless of
smoothing), so the same rbc_id always refers to the same physical cell in
both files. Edge status is computed once from the RAW contour and copied
into both files' class_code, rather than recomputed per variant.
"""
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as pipeline

SOURCE_DIR = Path(r"F:\Livo\Data - 2026\Rbc\7")
IMAGE_LIST_CSV = Path(r"F:\Livo\Data - 2026\Rbc\monolayer1_level7_fovs_detection_only.csv")

OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\rbc_query_deliverable_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROW_GROUP_SIZE = 4096
LEVEL1_MIN, LEVEL1_MAX = 5, 6
LEVEL_BUDGETS = {4: 8, 5: 16, 6: 24}
EPSILON_FRACTION = 0.05
EDGE_MARGIN = 2
CLASS_BIT_EDGE = 1 << 10
N_WORKERS = 7

import heapq


def importance_order(pts):
    n = len(pts)
    if n <= 2:
        return list(range(n))
    # the two globally farthest-apart points (the "diameter") are always a
    # pair of convex-hull vertices -- a standard computational-geometry fact
    # -- so restricting this search to hull vertices instead of all n points
    # is EXACT, not an approximation. This matters a lot for raw (no-smooth)
    # contours: at ~230 points/cell the naive all-pairs search is ~26k
    # comparisons, while a near-circular blob's hull is typically only
    # ~10-20 points (~100-200 comparisons) -- same result, ~100x less work.
    hull_idx = cv2.convexHull(pts.astype(np.float32), returnPoints=False).reshape(-1)
    best = (-1.0, int(hull_idx[0]), int(hull_idx[1]) if len(hull_idx) > 1 else int(hull_idx[0]))
    for hi in range(len(hull_idx)):
        i = int(hull_idx[hi])
        for hj in range(hi + 1, len(hull_idx)):
            j = int(hull_idx[hj])
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
            dists = [abs(seg[0] * (pts[k][1] - p1[1]) - seg[1] * (pts[k][0] - p1[0])) / seg_len for k in arc]
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


def lod_payload(pts, indices, min_x, min_y):
    return [
        {"point_index": int(idx), "dx": int(round(pts[idx][0])) - min_x, "dy": int(round(pts[idx][1])) - min_y}
        for idx in indices
    ]


def _zigzag(n):
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _uvarint(n):
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


def encode_exact_path(contour, min_x, min_y):
    relative = [(int(round(x)) - min_x, int(round(y)) - min_y) for x, y in contour]
    if any(dx < 0 or dy < 0 for dx, dy in relative):
        raise ValueError("contour point fell outside its own declared bounds")
    payload = bytearray()
    payload += _uvarint(len(relative))
    prev_x, prev_y = relative[0]
    payload += _uvarint(prev_x)
    payload += _uvarint(prev_y)
    for cx, cy in relative[1:]:
        payload += _uvarint(_zigzag(cx - prev_x))
        payload += _uvarint(_zigzag(cy - prev_y))
        prev_x, prev_y = cx, cy
    return bytes(payload)


SCHEMA = pa.schema([
    ("rbc_id", pa.int32()),
    ("min_x", pa.int32()),
    ("min_y", pa.int32()),
    ("width", pa.uint16()),
    ("height", pa.uint16()),
    ("class_code", pa.uint16()),
    ("level1_points", pa.list_(pa.struct([("point_index", pa.uint16()), ("dx", pa.uint16()), ("dy", pa.uint16())]))),
    ("level4_additions", pa.list_(pa.struct([("point_index", pa.uint16()), ("dx", pa.uint16()), ("dy", pa.uint16())]))),
    ("level5_additions", pa.list_(pa.struct([("point_index", pa.uint16()), ("dx", pa.uint16()), ("dy", pa.uint16())]))),
    ("level6_additions", pa.list_(pa.struct([("point_index", pa.uint16()), ("dx", pa.uint16()), ("dy", pa.uint16())]))),
    ("level7_path", pa.binary()),
])


def _build_cell_record(pts, tile_off_x, tile_off_y, class_code):
    """pts: Nx2 float array in full-tile-image coords. Returns (record_dict
    minus rbc_id, n_points). record's own min_x/min_y/width/height come from
    THIS pts array (raw or smoothed, whichever was passed)."""
    local_min_x = int(round(pts[:, 0].min()))
    local_min_y = int(round(pts[:, 1].min()))
    local_max_x = int(round(pts[:, 0].max()))
    local_max_y = int(round(pts[:, 1].max()))
    width = local_max_x - local_min_x
    height = local_max_y - local_min_y

    order = importance_order(pts)
    n = len(pts)
    l1n = level1_count(pts)
    c4 = min(n, LEVEL_BUDGETS[4])
    c5 = min(n, LEVEL_BUDGETS[5])
    c6 = min(n, LEVEL_BUDGETS[6])

    level7 = encode_exact_path(pts, local_min_x, local_min_y)

    record = {
        "rbc_id": 0,   # placeholder -- assigned sequentially (same value in both variants) by the main process
        "min_x": tile_off_x + local_min_x,
        "min_y": tile_off_y + local_min_y,
        "width": width,
        "height": height,
        "class_code": class_code,
        "level1_points": lod_payload(pts, order[:l1n], local_min_x, local_min_y),
        "level4_additions": lod_payload(pts, order[l1n:c4], local_min_x, local_min_y),
        "level5_additions": lod_payload(pts, order[c4:c5], local_min_x, local_min_y),
        "level6_additions": lod_payload(pts, order[c5:c6], local_min_x, local_min_y),
        "level7_path": level7,
    }
    return record, n


_worker_model = None


def _init_worker(_unused=None):
    global _worker_model
    _worker_model = pipeline.YOLO(str(pipeline.BEST_PT))


def _process_image_dual(task):
    """Runs box detection ONCE, then isolated_cell_contours TWICE (raw,
    smoothed) against the SAME boxes -- the two returned contour lists stay
    index-aligned to `boxes` (accept/reject and dedup/cluster decisions are
    made on the pixel mask, before smoothing, so they're identical either
    way). Cells are sorted by the ORIGINAL BOX's top-left corner -- a key
    that does not change between the raw and smoothed runs -- so the same
    cell lands at the same position in both output lists, and the main
    process can assign one rbc_id that means the same physical cell in both
    files. Edge status is computed once from the RAW contour's own bounds
    and baked into class_code for both variants, so it can't disagree
    between raw and smoothed just because smoothing nudged the shape a
    little past the edge margin."""
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

    pipeline.ENABLE_SMOOTHING = False
    raw_contours = pipeline.isolated_cell_contours(img, boxes, confidences)
    pipeline.ENABLE_SMOOTHING = True
    smooth_contours = pipeline.isolated_cell_contours(img, boxes, confidences)

    candidates = []
    for i, (raw_pts, smooth_pts) in enumerate(zip(raw_contours, smooth_contours)):
        if raw_pts is None or smooth_pts is None:
            continue
        bx1, by1, bx2, by2 = boxes[i]
        candidates.append((by1, bx1, raw_pts, smooth_pts))
    # stable key = the ORIGINAL BOX's top-left corner -- identical for both
    # the raw and smoothed run, so sort order (and therefore the eventual
    # rbc_id assignment) can never diverge between the two files
    candidates.sort(key=lambda c: (c[0], c[1]))

    cell_pairs = []
    for (_by1, _bx1, raw_pts, smooth_pts) in candidates:
        raw_pts = np.asarray(raw_pts, dtype=np.float64)
        smooth_pts = np.asarray(smooth_pts, dtype=np.float64)
        if len(raw_pts) < 3 or len(smooth_pts) < 3:
            continue

        r_min_x = int(round(raw_pts[:, 0].min()))
        r_min_y = int(round(raw_pts[:, 1].min()))
        r_max_x = int(round(raw_pts[:, 0].max()))
        r_max_y = int(round(raw_pts[:, 1].max()))
        edge = (
            r_min_x <= EDGE_MARGIN or r_min_y <= EDGE_MARGIN
            or r_max_x >= W - EDGE_MARGIN or r_max_y >= H - EDGE_MARGIN
        )
        class_code = CLASS_BIT_EDGE if edge else 0

        raw_record, n_raw = _build_cell_record(raw_pts, tile_off_x, tile_off_y, class_code)
        smooth_record, n_smooth = _build_cell_record(smooth_pts, tile_off_x, tile_off_y, class_code)
        cell_pairs.append((raw_record, n_raw, smooth_record, n_smooth))

    return idx, col, row, W, H, cell_pairs


LOD_BANDS = {
    "1": {"minimumPoints": 3, "maximumPoints": 6},
    "4": 8,
    "5": 16,
    "6": 24,
    "7": "exact-path",
}
CLASS_ENCODING = {
    "type": "packed-uint16",
    "fields": {
        "color": {"shift": 0, "bits": 2, "values": ["normal_color"]},
        "size": {"shift": 2, "bits": 3, "values": ["normal_size"]},
        "shape": {"shift": 5, "bits": 4, "values": ["normal_shape"]},
        "inclusion": {"shift": 9, "bits": 1, "values": ["no_inclusion"]},
    },
}
EDGE_DETECTION = {"field": "class_code", "bit": 10, "mask": CLASS_BIT_EDGE}
CELL_COLUMNS = ["rbc_id", "min_x", "min_y", "width", "height", "class_code"]
POINT_COLUMNS = ["rbc_id", "level1_points", "level4_additions", "level5_additions", "level6_additions", "level7_path"]


def write_output(rows, point_counts, total_points, parquet_file, manifest_file, label):
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, parquet_file, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    file_size = parquet_file.stat().st_size
    print(f"[{label}] Parquet -> {parquet_file} ({file_size:,} bytes)")

    pf = pq.ParquetFile(parquet_file)
    row_groups = []
    row_cursor = 0
    for gi in range(pf.num_row_groups):
        n_rows_in_group = pf.metadata.row_group(gi).num_rows
        start = row_cursor
        end = start + n_rows_in_group - 1
        row_groups.append({
            "rowGroup": gi,
            "minRbcId": rows[start]["rbc_id"],
            "maxRbcId": rows[end]["rbc_id"],
            "rows": n_rows_in_group,
            "points": sum(point_counts[start:end + 1]),
        })
        row_cursor += n_rows_in_group

    manifest = {
        "version": 5,
        "algorithm": "douglas-peucker",
        "rows": len(rows),
        "points": total_points,
        "coordinateLevel": 7,
        "coordinateEncoding": "rbc-relative",
        "lodLayout": "hybrid-incremental-exact-path",
        "datasetFile": parquet_file.name,
        "rowGroups": row_groups,
        "cellColumns": CELL_COLUMNS,
        "pointColumns": POINT_COLUMNS,
        "lodBands": LOD_BANDS,
        "classEncoding": CLASS_ENCODING,
        "edgeDetection": EDGE_DETECTION,
        "boundsEncoding": "min-width-height",
        "exactPathEncoding": "uvarint-count-origin-zigzag-delta-v1",
        "pointEncoding": "nested-delta-binary-packed-plus-zstd-exact-path",
        "pointLayout": "single-spatial-parquet-row-groups",
        "partial": False,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"[{label}] Manifest -> {manifest_file}")
    return file_size


def run_dual_pass(pool, img_paths):
    tasks = [(i, col, row, str(p)) for i, (col, row, p) in enumerate(img_paths)]
    results = [None] * len(img_paths)

    t_start = time.perf_counter()
    done = 0
    for idx, col, row, W, H, cell_pairs in pool.imap_unordered(_process_image_dual, tasks, chunksize=8):
        results[idx] = cell_pairs
        done += 1
        if done % 500 == 0 or done == len(tasks):
            elapsed_so_far = time.perf_counter() - t_start
            print(f"  {done:5d}/{len(tasks)} images done  elapsed={elapsed_so_far:7.1f}s")

    raw_rows, smooth_rows = [], []
    raw_points, smooth_points = [], []
    total_raw_points, total_smooth_points = 0, 0
    next_id = 0
    for cell_pairs in results:
        if cell_pairs is None:
            continue
        for raw_record, n_raw, smooth_record, n_smooth in cell_pairs:
            raw_record["rbc_id"] = next_id
            smooth_record["rbc_id"] = next_id
            raw_rows.append(raw_record)
            smooth_rows.append(smooth_record)
            raw_points.append(n_raw)
            smooth_points.append(n_smooth)
            total_raw_points += n_raw
            total_smooth_points += n_smooth
            next_id += 1

    elapsed = time.perf_counter() - t_start
    print(f"{len(raw_rows)} cells (rbc_id 0-{next_id - 1}), "
          f"{total_raw_points} raw points, {total_smooth_points} smoothed points, {elapsed:.1f}s")

    raw_size = write_output(raw_rows, raw_points, total_raw_points,
                             OUT_DIR / "rbc_query_no_smooth.parquet",
                             OUT_DIR / "rbc_query_no_smooth_manifest.json", "no-smooth")
    smooth_size = write_output(smooth_rows, smooth_points, total_smooth_points,
                                OUT_DIR / "rbc_query_with_smooth.parquet",
                                OUT_DIR / "rbc_query_with_smooth_manifest.json", "with-smooth")

    return elapsed, raw_size, smooth_size, len(raw_rows), total_raw_points, total_smooth_points


def main():
    img_paths = []
    with open(IMAGE_LIST_CSV, newline="") as f:
        for entry in csv.DictReader(f):
            col, row = int(entry["col"]), int(entry["row"])
            p = SOURCE_DIR / f"Img_{col}_{row}.jpg"
            if p.exists():
                img_paths.append((col, row, p))
    img_paths.sort(key=lambda t: (t[0], t[1]))
    print(f"{len(img_paths)} images from {IMAGE_LIST_CSV.name}  ({N_WORKERS} worker processes)")

    with multiprocessing.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        elapsed, raw_size, smooth_size, n_rows, raw_pts, smooth_pts = run_dual_pass(pool, img_paths)

    print("\n=== SUMMARY ===")
    print(f"time={elapsed:.1f}s  rows={n_rows}")
    print(f"no_smooth    size={raw_size:,} bytes  points={raw_pts}")
    print(f"with_smooth  size={smooth_size:,} bytes  points={smooth_pts}")


if __name__ == "__main__":
    main()
