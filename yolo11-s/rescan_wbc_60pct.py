"""
Re-scan every box in the (restored) training/val labels for WBC coverage,
using a 60% threshold -- the 80% cutoff missed some confirmed-real WBC boxes
(calibration samples reached as low as 0.65-0.71), so lowering it should
catch more of the genuine ones at the cost of pulling in more candidates to
manually review.
"""
from pathlib import Path
import cv2
import numpy as np

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
OUT = Path(r"F:\Livo\Data - 2026\Rbc\wbc_flagged_60pct")
OUT.mkdir(exist_ok=True)

WBC_HUE_MIN, WBC_HUE_MAX = 120, 155
WBC_SAT_MIN = 40
THRESHOLD = 0.60

flagged = []
n_images, n_boxes = 0, 0
for split in ("train", "val"):
    (OUT / split).mkdir(exist_ok=True)
    img_dir = DS / "images" / split
    lbl_dir = DS / "labels" / split
    for label_path in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / f"{label_path.stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        n_images += 1

        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        image_flags = []
        for line in lines:
            cls, cx, cy, w, h = [float(v) for v in line.split()]
            x1 = max(0, int((cx - w / 2) * W)); y1 = max(0, int((cy - h / 2) * H))
            x2 = min(W, int((cx + w / 2) * W)); y2 = min(H, int((cy + h / 2) * H))
            n_boxes += 1
            if x2 <= x1 or y2 <= y1:
                continue
            region = hsv[y1:y2, x1:x2]
            hue, sat = region[:, :, 0], region[:, :, 1]
            frac = ((hue >= WBC_HUE_MIN) & (hue <= WBC_HUE_MAX) & (sat >= WBC_SAT_MIN)).mean()
            if frac >= THRESHOLD:
                image_flags.append((x1, y1, x2, y2, round(float(frac), 3)))

        if image_flags:
            flagged.append((split, label_path.stem, image_flags))
            vis = img.copy()
            for line in lines:
                cls, cx, cy, w, h = [float(v) for v in line.split()]
                x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            for (x1, y1, x2, y2, frac) in image_flags:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(OUT / split / f"{label_path.stem}.jpg"), vis)

        if n_images % 500 == 0:
            print(f"  {n_images} images scanned, {len(flagged)} flagged so far...")

n_flagged_boxes = sum(len(f[2]) for f in flagged)
print(f"\nScanned {n_images} images, {n_boxes} boxes")
print(f"{len(flagged)} images flagged, {n_flagged_boxes} boxes flagged at >=60% WBC coverage")
print(f"Overlays saved -> {OUT}")

with open(OUT / "flagged_list.txt", "w") as f:
    for split, stem, boxes in flagged:
        f.write(f"{split}/{stem}: {boxes}\n")
