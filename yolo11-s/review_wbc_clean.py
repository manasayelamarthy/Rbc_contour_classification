"""
For images where clean_wbc_boxes.py actually removed something, draw the
CURRENT (kept) boxes in green and the REMOVED boxes (present in the backup,
gone from the cleaned labels) in red -- so it's directly visible whether the
removed boxes were real WBCs or legitimate RBCs that got wrongly stripped.
"""
from pathlib import Path
import cv2

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
BACKUP = DS / "labels_backup_before_wbc_clean"
OUT = Path(r"F:\Livo\Data - 2026\Rbc\wbc_clean_review")
OUT.mkdir(exist_ok=True)

N_SAMPLE = 40  # keep the review batch manageable; ask if more are wanted


def parse_lines(path):
    if not path.exists():
        return []
    return [l for l in path.read_text().splitlines() if l.strip()]


def to_box(line, W, H):
    cls, cx, cy, w, h = [float(v) for v in line.split()]
    x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
    return x1, y1, x2, y2


touched = []
for split in ("train", "val"):
    for backup_path in sorted((BACKUP / split).glob("*.txt")):
        current_path = DS / "labels" / split / backup_path.name
        backup_lines = set(parse_lines(backup_path))
        current_lines = set(parse_lines(current_path))
        removed = backup_lines - current_lines
        if removed:
            touched.append((split, backup_path.stem, backup_lines, current_lines, removed))

print(f"{len(touched)} images had boxes removed; saving overlays for {min(N_SAMPLE, len(touched))} of them")

# spread the sample across the whole touched list, not just the first N
step = max(1, len(touched) // N_SAMPLE)
sample = touched[::step][:N_SAMPLE]

for split, stem, backup_lines, current_lines, removed in sample:
    img_path = DS / "images" / split / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]
    for line in current_lines:
        x1, y1, x2, y2 = to_box(line, W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)   # kept -> green
    for line in removed:
        x1, y1, x2, y2 = to_box(line, W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)   # removed -> red
    cv2.imwrite(str(OUT / f"{split}_{stem}_review.jpg"), img)

print(f"Saved {len(sample)} overlays -> {OUT}")
print("Green = box kept after cleaning. Red = box removed by clean_wbc_boxes.py.")
