"""
Draw every box from the ORIGINAL (pre-cleaning) labels onto its image, for
manual review -- the automated WBC-vs-RBC color heuristic proved unreliable
(confirmed-real-WBC and confirmed-false-positive-RBC boxes have overlapping
HSV stats), so a human pass is needed to flag which images actually have a
box sitting on a WBC.
"""
from pathlib import Path
import cv2

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
LABELS_SRC = DS / "labels_backup_before_wbc_clean"   # original, uncleaned labels
OUT = Path(r"F:\Livo\Data - 2026\Rbc\train_overlays_for_wbc_flagging")
OUT.mkdir(exist_ok=True)


def draw_boxes(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = [float(v) for v in line.split()]
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.imwrite(str(out_path), img)
    return True


n_done = 0
for split in ("train", "val"):
    (OUT / split).mkdir(exist_ok=True)
    img_dir = DS / "images" / split
    for label_path in sorted((LABELS_SRC / split).glob("*.txt")):
        img_path = img_dir / f"{label_path.stem}.jpg"
        out_path = OUT / split / f"{label_path.stem}.jpg"
        if draw_boxes(img_path, label_path, out_path):
            n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done} done...")

print(f"\nSaved {n_done} overlay images -> {OUT}")
print("Boxes drawn from the ORIGINAL (pre-cleaning) labels -- go through these")
print("and note which images have a box sitting on a genuine WBC.")
