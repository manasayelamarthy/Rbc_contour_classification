"""
Run the box+CV contour pipeline on the x=23,24 / y=0-127 test slice from the
master tile pool, saving box + contour overlays into their own folder.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(r"F:\Livo\Data - 2026\Rbc\images")))
import infer_contours_outer_boundary as pipeline

SOURCE_DIR = Path(r"F:\Livo\Data - 2026\Rbc\7")
TEST_X = [23, 24]
TEST_Y = range(128)

OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\rbc_contour_output_x23_24")
BOX_DIR = OUT_DIR / "overlays_boxes"
CONTOUR_DIR = OUT_DIR / "overlays_contours"
BOX_DIR.mkdir(parents=True, exist_ok=True)
CONTOUR_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = pipeline.YOLO(str(pipeline.BEST_PT))
    print(f"Model: {pipeline.BEST_PT}")

    img_paths = []
    for y in TEST_Y:
        for x in TEST_X:
            p = SOURCE_DIR / f"Img_{y}_{x}.jpg"
            if p.exists():
                img_paths.append(p)
    print(f"{len(img_paths)} images (x in {TEST_X}, y 0-{max(TEST_Y)}) -> {OUT_DIR}")

    total_boxes, total_contours = 0, 0
    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx:3d}] SKIP (unreadable): {img_path.name}")
            continue

        dets, timing = pipeline.detect_and_contour(model, img)
        n_contours = sum(1 for d in dets if "contour" in d)
        total_boxes += len(dets)
        total_contours += n_contours

        box_vis = pipeline.draw_boxes(img, dets)
        cv2.imwrite(str(BOX_DIR / (img_path.stem + "_boxes.jpg")), box_vis)

        contour_vis = pipeline.draw_contours(img, dets)
        cv2.imwrite(str(CONTOUR_DIR / (img_path.stem + "_contours.jpg")), contour_vis)

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"  [{idx:3d}/{len(img_paths)}] {img_path.name:<16s}  boxes={len(dets):3d}  contours={n_contours:3d}")

    print(f"\n{len(img_paths)} images  total_boxes={total_boxes}  total_contours={total_contours}")
    print(f"Box overlays     -> {BOX_DIR}")
    print(f"Contour overlays -> {CONTOUR_DIR}")


if __name__ == "__main__":
    main()
