"""
Real validation of the pallor-blob classification method (Normal / Spherocyte
/ Stomatocyte / Target cell) against the real labeled RBC morphology dataset
(Zenodo 10.5281/zenodo.14936017).

These images are pre-cropped single-cell images from a DIFFERENT source than
our own tiles, with different staining/color calibration -- our hue-
restricted whole-image mask doesn't transfer, so segmentation here uses a
simple Otsu-on-saturation mask (no hue restriction), reasonable since each
image is already isolated to one cell with minimal background.
"""
import random
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_pallor_classification import analyze_pallor

DATASET_DIR = Path(__file__).parent / "rbc_reference_dataset" / "RBC" / "train"

# map dataset folder names -> our classifier's output categories
CLASS_MAP = {
    "normal": "Normal",
    "Spherocyte": "Spherocyte",
    "Stomatocyte": "Stomatocyte",
    "Target Cell": "Target cell",
}
N_PER_CLASS = 60


def simple_cell_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, fg = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return fg, hsv


def get_main_cell_contour(img):
    fg, hsv = simple_cell_mask(img)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None, hsv
    # pick the component closest to the image center (that's the labeled cell;
    # small fragments near the border are usually neighboring cells/noise)
    h, w = fg.shape
    cx0, cy0 = w / 2, h / 2
    best_lbl, best_d = None, None
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        ccx, ccy = centroids[lbl]
        d = (ccx - cx0) ** 2 + (ccy - cy0) ** 2
        if best_d is None or d < best_d:
            best_d, best_lbl = d, lbl
    if best_lbl is None:
        return None, hsv
    mask = (labels == best_lbl).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, hsv
    best = max(contours, key=cv2.contourArea)
    return best.reshape(-1, 2).astype(np.float32), hsv


def main():
    random.seed(0)
    confusion = {true: Counter() for true in CLASS_MAP.values()}
    skipped = Counter()

    for folder_name, true_label in CLASS_MAP.items():
        class_dir = DATASET_DIR / folder_name
        # use only ORIGINAL images (skip _blurred/_flipped/_rotated augmentations)
        # so we're testing on distinct real photos, not augmented duplicates
        all_files = sorted(class_dir.glob("*"))
        originals = [f for f in all_files if "_" not in f.stem]
        sample = originals[:N_PER_CLASS] if len(originals) >= N_PER_CLASS else originals
        print(f"{folder_name}: testing {len(sample)} images")

        for f in sample:
            img = cv2.imread(str(f))
            if img is None:
                skipped[true_label] += 1
                continue
            pts, hsv = get_main_cell_contour(img)
            if pts is None or len(pts) < 5:
                skipped[true_label] += 1
                continue
            result = analyze_pallor(pts, hsv)
            if result is None:
                skipped[true_label] += 1
                continue
            confusion[true_label][result["category"]] += 1

    print("\n=== CONFUSION MATRIX (rows=true label, cols=predicted) ===")
    pred_cats = sorted(set(c for counts in confusion.values() for c in counts) | set(CLASS_MAP.values()))
    label = "TRUE/PRED"
    header = f"{label:<14s}" + "".join(f"{c:>14s}" for c in pred_cats)
    print(header)
    total_correct, total_n = 0, 0
    for true_label, counts in confusion.items():
        row_total = sum(counts.values())
        row = f"{true_label:<14s}" + "".join(f"{counts.get(c, 0):>14d}" for c in pred_cats)
        print(row)
        total_correct += counts.get(true_label, 0)
        total_n += row_total
    print(f"\nOverall accuracy: {total_correct}/{total_n} = {100*total_correct/total_n:.1f}%")
    print("Skipped (segmentation failed):", dict(skipped))


if __name__ == "__main__":
    main()
