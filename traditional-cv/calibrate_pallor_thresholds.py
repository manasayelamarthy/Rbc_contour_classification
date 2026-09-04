"""
Gathers raw pallor-blob features (not yet thresholded) across the 4 real
labeled classes, at several PALLOR_FRAC sensitivity settings, so we can pick
thresholds from actual data instead of guessing.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_pallor_classification import DATASET_DIR, CLASS_MAP, get_main_cell_contour

N_PER_CLASS = 60
FRACS_TO_TEST = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def pallor_area_frac_at(pts, hsv, pallor_frac):
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
    pallor_thresh = own_median_sat * pallor_frac
    pallor_mask = (sat_crop < pallor_thresh) & cell_region
    cell_area = cell_region.sum()
    return pallor_mask.sum() / cell_area if cell_area else 0


def main():
    data = {label: {f: [] for f in FRACS_TO_TEST} for label in CLASS_MAP.values()}

    for folder_name, true_label in CLASS_MAP.items():
        class_dir = DATASET_DIR / folder_name
        all_files = sorted(class_dir.glob("*"))
        originals = [f for f in all_files if "_" not in f.stem]
        sample = originals[:N_PER_CLASS]

        for f in sample:
            img = cv2.imread(str(f))
            if img is None:
                continue
            pts, hsv = get_main_cell_contour(img)
            if pts is None or len(pts) < 5:
                continue
            for frac in FRACS_TO_TEST:
                v = pallor_area_frac_at(pts, hsv, frac)
                if v is not None:
                    data[true_label][frac].append(v)

    print(f"{'class':<14s} " + "".join(f"frac={f:<10.2f}" for f in FRACS_TO_TEST))
    for label in CLASS_MAP.values():
        row = f"{label:<14s} "
        for frac in FRACS_TO_TEST:
            vals = data[label][frac]
            med = np.median(vals) if vals else float("nan")
            row += f"{med:<15.4f}"
        print(row)

    print("\n--- percentiles at frac=0.9 (most informative column, printed above) ---")
    for label in CLASS_MAP.values():
        vals = data[label][0.9]
        if vals:
            p = np.percentile(vals, [10, 25, 50, 75, 90])
            print(f"{label:<14s} p10={p[0]:.4f} p25={p[1]:.4f} p50={p[2]:.4f} p75={p[3]:.4f} p90={p[4]:.4f}")


if __name__ == "__main__":
    main()
