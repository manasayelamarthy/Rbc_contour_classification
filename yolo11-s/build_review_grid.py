"""
Build a compact visual grid of every box flagged at >=60% WBC coverage, so
they can all be judged efficiently in a handful of montage images instead of
one-by-one. Each thumbnail is labeled with an index; a companion index file
maps index -> (split, image stem, box coords) for the removal step.
"""
import ast
from pathlib import Path
import cv2
import numpy as np

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
FLAGGED_LIST = Path(r"F:\Livo\Data - 2026\Rbc\wbc_flagged_60pct\flagged_list.txt")
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\wbc_review_grid")
OUT_DIR.mkdir(exist_ok=True)

THUMB = 110
PAD = 8
COLS = 14
PER_PAGE = COLS * 16   # 224 thumbnails/page

entries = []   # (split, stem, x1,y1,x2,y2, frac)
for line in FLAGGED_LIST.read_text().splitlines():
    if not line.strip():
        continue
    key, rest = line.split(":", 1)
    split, stem = key.split("/")
    boxes = ast.literal_eval(rest.strip())
    for (x1, y1, x2, y2, frac) in boxes:
        entries.append((split, stem, x1, y1, x2, y2, frac))

print(f"{len(entries)} boxes to render")

index_lines = []
thumbs = []
for idx, (split, stem, x1, y1, x2, y2, frac) in enumerate(entries):
    img_path = DS / "images" / split / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        thumbs.append(np.zeros((THUMB, THUMB, 3), dtype=np.uint8))
        index_lines.append(f"{idx}: {split}/{stem} ({x1},{y1},{x2},{y2}) frac={frac} [UNREADABLE]")
        continue
    H, W = img.shape[:2]
    cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
    cx2, cy2 = min(W, x2 + PAD), min(H, y2 + PAD)
    crop = img[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    cv2.rectangle(crop, (PAD, PAD), (THUMB - PAD, THUMB - PAD), (0, 255, 0), 1)
    cv2.putText(crop, str(idx), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    thumbs.append(crop)
    index_lines.append(f"{idx}: {split}/{stem} ({x1},{y1},{x2},{y2}) frac={frac}")

(OUT_DIR / "index.txt").write_text("\n".join(index_lines))

for page_start in range(0, len(thumbs), PER_PAGE):
    page_thumbs = thumbs[page_start:page_start + PER_PAGE]
    rows = (len(page_thumbs) + COLS - 1) // COLS
    grid = np.full((rows * THUMB, COLS * THUMB, 3), 30, dtype=np.uint8)
    for i, thumb in enumerate(page_thumbs):
        r, c = divmod(i, COLS)
        grid[r*THUMB:(r+1)*THUMB, c*THUMB:(c+1)*THUMB] = thumb
    page_num = page_start // PER_PAGE
    cv2.imwrite(str(OUT_DIR / f"grid_page{page_num}.png"), grid)
    print(f"page {page_num}: indices {page_start}-{page_start+len(page_thumbs)-1}")

print(f"\nSaved grid pages + index.txt -> {OUT_DIR}")
