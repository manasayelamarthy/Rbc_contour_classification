"""
Build the rbc_query.parquet + rbc_query_manifest.json deliverable, matching
the web developer's exact spec:

- Columns: rbc_id:int32, min_x:int32, min_y:int32, width:uint16, height:uint16,
  class_code:uint16, four LOD list columns of {point_index:uint16, dx:uint16,
  dy:uint16}, level7_path:binary.
- rbc_id globally unique, spatially ordered, never restarts per image.
- min_x/min_y are SLIDE-GLOBAL integer coordinates (tile index * tile size +
  local pixel position) -- not per-tile-local.
- LOD point coordinates (dx, dy) and level7_path are RBC-relative (offset
  from that cell's own min_x/min_y), always non-negative.
- level7_path: uvarint-count-origin-zigzag-delta-v1 -- unsigned-varint point
  count, unsigned-varint origin x, unsigned-varint origin y, then
  zigzag-varint (dx,dy) deltas for every remaining point. Contains the
  COMPLETE exact contour (matches the pipeline's post-smooth/simplify
  polygon, not the raw ~200pt findContours output).
- class_code: packed uint16, bit 10 (1024) reserved for "edge cell" (box
  touches its source tile's crop boundary, so the true shape may continue
  into a neighboring tile).
- Zstandard compression, spatial row groups of ~4096 cells.
"""
import json
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
TEST_X = [23, 24]
TEST_Y = range(128)

OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\rbc_query_deliverable")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_FILE = OUT_DIR / "rbc_query.parquet"
MANIFEST_FILE = OUT_DIR / "rbc_query_manifest.json"

ROW_GROUP_SIZE = 4096
LEVEL1_MIN, LEVEL1_MAX = 5, 6
LEVEL_BUDGETS = {4: 8, 5: 16, 6: 24}
EPSILON_FRACTION = 0.05
EDGE_MARGIN = 2   # px tolerance for "touches the tile boundary"
CLASS_BIT_EDGE = 1 << 10   # bit 10 = 1024


# ---- Douglas-Peucker-style importance ranking (unchanged from the pilot) ----
import heapq


def importance_order(pts):
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


def lod_payload(pts, indices, min_x, min_y):
    """RBC-relative (dx,dy) for each point, plus its index in the full contour."""
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
    """uvarint-count-origin-zigzag-delta-v1 -- matches the reference spec
    exactly: unsigned-varint count, unsigned-varint origin (x,y) relative to
    min_x/min_y, then zigzag-varint (dx,dy) deltas for every remaining point."""
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


def main():
    model = pipeline.YOLO(str(pipeline.BEST_PT))
    print(f"Model: {pipeline.BEST_PT}")

    img_paths = []
    for y in TEST_Y:
        for x in TEST_X:
            p = SOURCE_DIR / f"Img_{y}_{x}.jpg"
            if p.exists():
                img_paths.append((y, x, p))
    img_paths.sort(key=lambda t: (t[0], t[1]))   # spatial order: row-major (y, x)
    print(f"{len(img_paths)} images (x in {TEST_X}, y 0-{max(TEST_Y)})")

    if img_paths:
        _ = pipeline.detect_and_contour(model, cv2.imread(str(img_paths[0][2])))

    rows = []
    point_counts = []
    next_id = 0
    total_points = 0
    t_start = time.perf_counter()
    for idx, (ty, tx, img_path) in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        # filename is Img_{col}_{row}.jpg -- ty is the loop's first filename
        # number (col), tx is the second (row). global_x uses col, global_y
        # uses row -- NOT the reverse (that was the bug: it turned the
        # horizontal col=0-127/row=23-24 slice into a vertical strip).
        tile_off_x, tile_off_y = ty * W, tx * H

        dets, _timing = pipeline.detect_and_contour(model, img)

        cells = []
        for det in dets:
            if "contour" not in det:
                continue
            pts = np.array(det["contour"], dtype=np.float64)
            if len(pts) < 3:
                continue
            local_min_x = int(round(pts[:, 0].min()))
            local_min_y = int(round(pts[:, 1].min()))
            local_max_x = int(round(pts[:, 0].max()))
            local_max_y = int(round(pts[:, 1].max()))
            cells.append((local_min_y, local_min_x, pts, local_min_x, local_min_y, local_max_x, local_max_y))
        cells.sort(key=lambda c: (c[0], c[1]))   # spatial order within tile: row-major

        for (_sy, _sx, pts, local_min_x, local_min_y, local_max_x, local_max_y) in cells:
            width = local_max_x - local_min_x
            height = local_max_y - local_min_y

            order = importance_order(pts)
            n = len(pts)
            l1n = level1_count(pts)
            c4 = min(n, LEVEL_BUDGETS[4])
            c5 = min(n, LEVEL_BUDGETS[5])
            c6 = min(n, LEVEL_BUDGETS[6])

            edge = (
                local_min_x <= EDGE_MARGIN or local_min_y <= EDGE_MARGIN
                or local_max_x >= W - EDGE_MARGIN or local_max_y >= H - EDGE_MARGIN
            )
            class_code = CLASS_BIT_EDGE if edge else 0

            level7 = encode_exact_path(pts, local_min_x, local_min_y)
            total_points += n
            point_counts.append(n)

            rows.append({
                "rbc_id": next_id,
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
            })
            next_id += 1

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"  [{idx:3d}/{len(img_paths)}] {img_path.name:<16s}  total_cells={len(rows)}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{len(rows)} cells, {total_points} total exact-contour points, {elapsed:.1f}s")

    # ---- build the pyarrow table with exact types ----
    lod_struct = pa.struct([
        ("point_index", pa.uint16()),
        ("dx", pa.uint16()),
        ("dy", pa.uint16()),
    ])
    lod_list = pa.list_(lod_struct)
    schema = pa.schema([
        ("rbc_id", pa.int32()),
        ("min_x", pa.int32()),
        ("min_y", pa.int32()),
        ("width", pa.uint16()),
        ("height", pa.uint16()),
        ("class_code", pa.uint16()),
        ("level1_points", lod_list),
        ("level4_additions", lod_list),
        ("level5_additions", lod_list),
        ("level6_additions", lod_list),
        ("level7_path", pa.binary()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)

    pq.write_table(table, PARQUET_FILE, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    file_size = PARQUET_FILE.stat().st_size
    print(f"Parquet -> {PARQUET_FILE} ({file_size:,} bytes)")

    # read back the ACTUAL physical row groups from the written file so the
    # manifest matches the real parquet structure exactly
    pf = pq.ParquetFile(PARQUET_FILE)
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
        "datasetFile": "rbc_query.parquet",
        "rowGroups": row_groups,
        "boundsEncoding": "min-width-height",
        "exactPathEncoding": "uvarint-count-origin-zigzag-delta-v1",
        "pointEncoding": "nested-delta-binary-packed-plus-zstd-exact-path",
        "pointLayout": "single-spatial-parquet-row-groups",
        "partial": False,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest -> {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
