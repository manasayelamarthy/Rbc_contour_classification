"""
Runs the pure classical (no-model) contour pipeline on the x=23,24 / y=0-127
tile slice, saving contour-only overlays. Same tile slice/convention as
yolo11-s\\run_contours_x23_24.py, for direct comparison.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype_classical_only import classical_isolated_cell_contours

SOURCE_DIR = Path(r"F:\Livo\Data - 2026\Rbc\others\7")
TEST_X = [23, 24]
TEST_Y = range(128)

OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\traditional-cv\rbc_contour_output_x23_24")
CONTOUR_DIR = OUT_DIR / "overlays_contours"
CONTOUR_DIR.mkdir(parents=True, exist_ok=True)

MIN_AREA, MAX_AREA, MIN_CIRC = 3500, 7500, 0.75
CONTOUR_COLOR = (255, 0, 255)


def main():
    img_paths = []
    for y in TEST_Y:
        for x in TEST_X:
            p = SOURCE_DIR / f"Img_{y}_{x}.jpg"
            if p.exists():
                img_paths.append(p)
    print(f"{len(img_paths)} images (x in {TEST_X}, y 0-{max(TEST_Y)}) -> {OUT_DIR}")

    total_cells = 0
    t_start = time.perf_counter()
    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx:3d}] SKIP (unreadable): {img_path.name}")
            continue

        t0 = time.perf_counter()
        contours = classical_isolated_cell_contours(img, MIN_AREA, MAX_AREA, MIN_CIRC)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_cells += len(contours)

        vis = img.copy()
        for pts in contours:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], True, CONTOUR_COLOR, 1, cv2.LINE_AA)
        cv2.imwrite(str(CONTOUR_DIR / (img_path.stem + "_contours.jpg")), vis)

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"  [{idx:3d}/{len(img_paths)}] {img_path.name:<16s}  cells={len(contours):3d}  time={elapsed_ms:5.1f}ms")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n{len(img_paths)} images  total_cells={total_cells}  total_time={total_elapsed:.1f}s  "
          f"avg={total_elapsed/len(img_paths)*1000:.1f}ms/image")
    print(f"Contour overlays -> {CONTOUR_DIR}")


if __name__ == "__main__":
    main()
