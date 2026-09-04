"""
Copy the overlay images (original, uncleaned boxes drawn) for just the 574
images where clean_wbc_boxes.py actually found and removed something -- the
candidate set my heuristic flagged as possibly-WBC -- into their own folder
for focused manual review.
"""
import shutil
from pathlib import Path

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
BACKUP = DS / "labels_backup_before_wbc_clean"
OVERLAYS = Path(r"F:\Livo\Data - 2026\Rbc\train_overlays_for_wbc_flagging")
OUT = Path(r"F:\Livo\Data - 2026\Rbc\wbc_flagged_candidates")
OUT.mkdir(exist_ok=True)


def parse_lines(p):
    return set(l for l in p.read_text().splitlines() if l.strip()) if p.exists() else set()


n_copied = 0
for split in ("train", "val"):
    (OUT / split).mkdir(exist_ok=True)
    for backup_path in sorted((BACKUP / split).glob("*.txt")):
        current_path = DS / "labels" / split / backup_path.name
        removed = parse_lines(backup_path) - parse_lines(current_path)
        if removed:
            src = OVERLAYS / split / f"{backup_path.stem}.jpg"
            if src.exists():
                shutil.copy2(src, OUT / split / src.name)
                n_copied += 1

print(f"Copied {n_copied} flagged-candidate overlay images -> {OUT}")
