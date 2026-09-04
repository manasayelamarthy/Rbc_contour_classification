"""
Interactive box annotation tool for the RBC training dataset.

- Left-click and drag: draw a NEW box (for an RBC that was left untouched /
  never got a box).
- Right-click on an existing box: DELETE it (for a box wrongly sitting on a
  WBC or platelet). If boxes overlap under the cursor, deletes the smallest
  (most specific) one.
- Changes save automatically to the image's YOLO label .txt file whenever
  you move to another image, or immediately with 's'.

Keys:
  n / d / Right Arrow  -> next image (saves first if changed)
  p / a / Left Arrow   -> previous image (saves first if changed)
  s                    -> save now, without moving
  q / Esc              -> save (if changed) and quit
  j                     -> jump to image by typing its number in the console

Run (needs a real display -- run it yourself in a normal terminal/desktop
session, not through a remote/background shell):
    "F:\\envs\\rbc\\python.exe" "F:\\Livo\\Data - 2026\\Rbc\\annotate_tool.py"
"""
from pathlib import Path

import cv2
import numpy as np

# ---- Config: which images to go through ------------------------------------
DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")
SPLITS = ["train", "val"]          # edit to ["train"] or ["val"] to narrow it
START_INDEX = 0                    # edit to resume partway through

MAX_DISPLAY_W, MAX_DISPLAY_H = 1500, 950
MIN_BOX_PX = 4                     # ignore accidental tiny drags/clicks


def label_path_for(img_path, split):
    return DS / "labels" / split / f"{img_path.stem}.txt"


def load_boxes(lbl_path, W, H):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            _cls, cx, cy, w, h = [float(v) for v in line.split()]
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            boxes.append([x1, y1, x2, y2])
    return boxes


def save_boxes(lbl_path, boxes, W, H):
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))


class Annotator:
    def __init__(self, entries):
        self.entries = entries   # list of (split, img_path)
        self.idx = START_INDEX
        self.drawing = False
        self.start_pt = None
        self.cur_pt = None
        self.dirty = False
        self.img = None
        self.boxes = []
        self.scale = 1.0
        self.load(self.idx)

    def load(self, idx):
        self.idx = idx % len(self.entries)
        split, img_path = self.entries[self.idx]
        self.split = split
        self.img_path = img_path
        self.img = cv2.imread(str(img_path))
        H, W = self.img.shape[:2]
        self.W, self.H = W, H
        self.scale = min(MAX_DISPLAY_W / W, MAX_DISPLAY_H / H, 1.0)
        self.disp_w, self.disp_h = int(W * self.scale), int(H * self.scale)
        self.lbl_path = label_path_for(img_path, split)
        self.boxes = load_boxes(self.lbl_path, W, H)
        self.dirty = False

    def save(self):
        if self.img is None:
            return
        save_boxes(self.lbl_path, self.boxes, self.W, self.H)
        self.dirty = False

    def render(self):
        disp = cv2.resize(self.img, (self.disp_w, self.disp_h))
        for (x1, y1, x2, y2) in self.boxes:
            p1 = (int(x1 * self.scale), int(y1 * self.scale))
            p2 = (int(x2 * self.scale), int(y2 * self.scale))
            cv2.rectangle(disp, p1, p2, (0, 200, 0), 1)
        if self.drawing and self.start_pt and self.cur_pt:
            cv2.rectangle(disp, self.start_pt, self.cur_pt, (0, 0, 255), 1)
        status = (f"[{self.idx + 1}/{len(self.entries)}] {self.split}/{self.img_path.name}"
                  f"  boxes={len(self.boxes)}" + ("  *unsaved*" if self.dirty else ""))
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 22), (30, 30, 30), -1)
        cv2.putText(disp, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return disp

    def mouse_cb(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.cur_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cur_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_pt
                x2, y2 = x, y
                ox1, oy1 = min(x1, x2) / self.scale, min(y1, y2) / self.scale
                ox2, oy2 = max(x1, x2) / self.scale, max(y1, y2) / self.scale
                if (ox2 - ox1) * self.scale > MIN_BOX_PX and (oy2 - oy1) * self.scale > MIN_BOX_PX:
                    self.boxes.append([ox1, oy1, ox2, oy2])
                    self.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            ox, oy = x / self.scale, y / self.scale
            best_i, best_area = None, None
            for i, (x1, y1, x2, y2) in enumerate(self.boxes):
                if x1 <= ox <= x2 and y1 <= oy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best_area is None or area < best_area:
                        best_i, best_area = i, area
            if best_i is not None:
                del self.boxes[best_i]
                self.dirty = True


def main():
    entries = []
    for split in SPLITS:
        img_dir = DS / "images" / split
        for img_path in sorted(img_dir.glob("*.jpg")):
            entries.append((split, img_path))
    print(f"{len(entries)} images loaded ({', '.join(SPLITS)})")
    print("Left-drag = new box | Right-click a box = delete it | n/p = next/prev | s = save | q = save+quit")

    ann = Annotator(entries)
    cv2.namedWindow("annotator")
    cv2.setMouseCallback("annotator", ann.mouse_cb)

    while True:
        cv2.imshow("annotator", ann.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("n"), ord("d")):
            if ann.dirty:
                ann.save()
            ann.load(ann.idx + 1)
        elif key in (ord("p"), ord("a")):
            if ann.dirty:
                ann.save()
            ann.load(ann.idx - 1)
        elif key == ord("s"):
            ann.save()
            print(f"saved {ann.split}/{ann.img_path.name}")
        elif key == ord("j"):
            try:
                target = int(input(f"jump to image # (1-{len(entries)}): ")) - 1
            except ValueError:
                target = None
            if target is not None:
                if ann.dirty:
                    ann.save()
                ann.load(target)
        elif key in (ord("q"), 27):
            if ann.dirty:
                ann.save()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
