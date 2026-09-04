"""
Test: can we actually separate Normal / Spherocyte / Stomatocyte / Target-cell
using the pallor-blob method, on one real image? Honest empirical check --
prints per-cell numbers and saves crops from each predicted category so we
can visually verify, not just trust the numbers.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype_classical_only import classical_isolated_cell_contours

PALLOR_FRAC = 0.75   # a pixel counts as "pallor" if its saturation is below
                      # this fraction of the cell's own median saturation
MIN_PALLOR_AREA_FRAC = 0.03   # below this fraction of cell area -> "no pallor"
STOMATOCYTE_ASPECT = 1.8      # pallor blob aspect ratio above this -> slit-shaped


def analyze_pallor(pts, hsv):
    c = pts.reshape(-1, 1, 2).astype(np.int32)
    bx, by, bw, bh = cv2.boundingRect(c)
    cell_mask = np.zeros((bh, bw), np.uint8)
    cv2.drawContours(cell_mask, [c - [bx, by]], -1, 255, -1)
    cell_region = cell_mask > 0

    sat_crop = hsv[by:by + bh, bx:bx + bw, 1].astype(np.float32)
    cell_sat_vals = sat_crop[cell_region]
    if cell_sat_vals.size == 0:
        return None
    own_median_sat = np.median(cell_sat_vals)

    pallor_thresh = own_median_sat * PALLOR_FRAC
    pallor_mask = ((sat_crop < pallor_thresh) & cell_region).astype(np.uint8) * 255

    cell_area = cell_region.sum()
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pallor_mask, connectivity=8)
    if n_labels <= 1:
        return {"category": "Spherocyte", "pallor_area_frac": 0.0, "aspect": None,
                "bbox": (bx, by, bw, bh), "own_median_sat": own_median_sat}

    # largest pallor component (excluding background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = np.argmax(areas) + 1
    pallor_area = stats[biggest, cv2.CC_STAT_AREA]
    pallor_area_frac = pallor_area / cell_area if cell_area else 0

    if pallor_area_frac < MIN_PALLOR_AREA_FRAC:
        return {"category": "Spherocyte", "pallor_area_frac": pallor_area_frac, "aspect": None,
                "bbox": (bx, by, bw, bh), "own_median_sat": own_median_sat}

    blob_mask = (labels == biggest).astype(np.uint8)
    blob_pts = cv2.findNonZero(blob_mask)
    if blob_pts is None or len(blob_pts) < 5:
        aspect = None
    else:
        (_, _), (minor, major), _ = cv2.fitEllipse(blob_pts)
        aspect = max(major, minor) / max(min(major, minor), 1e-6)

    if aspect is not None and aspect >= STOMATOCYTE_ASPECT:
        category = "Stomatocyte"
    else:
        # check for a darker spot enclosed WITHIN the pallor blob (target-cell ring)
        blob_bx, blob_by, blob_bw, blob_bh = stats[biggest, cv2.CC_STAT_LEFT], stats[biggest, cv2.CC_STAT_TOP], \
            stats[biggest, cv2.CC_STAT_WIDTH], stats[biggest, cv2.CC_STAT_HEIGHT]
        inner_sat = sat_crop[blob_by:blob_by + blob_bh, blob_bx:blob_bx + blob_bw]
        inner_blob_mask = blob_mask[blob_by:blob_by + blob_bh, blob_bx:blob_bx + blob_bw] > 0
        # a "dark spot inside the pale ring" would show up as non-pallor pixels
        # (holes) fully surrounded by the pallor blob -- check via inverse flood fill
        inv = (~inner_blob_mask).astype(np.uint8) * 255
        n2, lbl2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        has_enclosed_hole = False
        for lbl_id in range(1, n2):
            x2, y2, w2, h2, a2 = stats2[lbl_id]
            touches_border = (x2 == 0 or y2 == 0 or x2 + w2 >= inv.shape[1] or y2 + h2 >= inv.shape[0])
            if not touches_border and a2 >= 3:
                has_enclosed_hole = True
                break
        category = "Target cell" if has_enclosed_hole else "Normal"

    return {"category": category, "pallor_area_frac": pallor_area_frac, "aspect": aspect,
            "bbox": (bx, by, bw, bh), "own_median_sat": own_median_sat}


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Livo\Data - 2026\Rbc\others\7\Img_7_13.jpg"
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    contours = classical_isolated_cell_contours(img, min_area=3500, max_area=7500, min_circularity=0.85)
    print(f"{len(contours)} round/normal-circularity cells found\n")

    results = []
    for pts in contours:
        r = analyze_pallor(pts, hsv)
        if r is not None:
            results.append((pts, r))

    from collections import Counter
    dist = Counter(r["category"] for _, r in results)
    print("Category distribution:", dict(dist))
    print()

    out_dir = Path(__file__).parent / "pallor_test_crops"
    out_dir.mkdir(exist_ok=True)
    saved_per_cat = Counter()
    for pts, r in results:
        cat = r["category"]
        if saved_per_cat[cat] >= 4:
            continue
        bx, by, bw, bh = r["bbox"]
        pad = 10
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = bx + bw + pad, by + bh + pad
        crop = img[y1:y2, x1:x2].copy()
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2) - [x1, y1]
        cv2.polylines(crop, [poly], True, (255, 0, 255), 1, cv2.LINE_AA)
        crop_big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        idx = saved_per_cat[cat]
        safe_cat = cat.replace(" ", "_")
        cv2.imwrite(str(out_dir / f"{safe_cat}_{idx}.jpg"), crop_big)
        saved_per_cat[cat] += 1

    print("Saved example crops per category to", out_dir)
    for cat, n in saved_per_cat.items():
        print(f"  {cat}: {n} examples saved")


if __name__ == "__main__":
    main()
