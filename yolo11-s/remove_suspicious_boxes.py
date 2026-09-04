from pathlib import Path
import cv2

DS = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded")

# (split, stem, x1,y1,x2,y2) for each box judged suspicious (real WBC material)
SUSPICIOUS = [
    ("train", "Img_101_24", 1169, 0, 1224, 50),
    ("train", "Img_103_24", 532, 341, 557, 368),
    ("train", "Img_107_35", 2, 369, 75, 447),
    ("train", "Img_109_32", 521, 875, 588, 959),
    ("train", "Img_115_65", 966, 0, 1101, 39),
    ("train", "Img_11_20", 1501, 1072, 1530, 1093),
    ("train", "Img_12_32", 445, 961, 466, 981),
    ("train", "Img_14_35", 1, 633, 18, 653),
    ("train", "Img_15_11", 1189, 1083, 1206, 1094),
    ("train", "Img_24_32", 227, 694, 256, 719),
    ("train", "Img_56_65", 824, 981, 847, 1005),
    ("train", "Img_59_62", 9, 538, 38, 569),
    ("train", "Img_5_10", 570, 443, 587, 462),
    ("train", "Img_67_68", 1435, 1000, 1500, 1073),
    ("train", "Img_68_60", 1561, 927, 1592, 954),
    ("train", "Img_70_71", 1484, 301, 1555, 383),
    ("train", "Img_77_60", 1534, 185, 1597, 263),
    ("train", "Img_77_67", 1583, 239, 1598, 260),
    ("train", "Img_82_45", 1036, 97, 1099, 185),
    ("train", "Img_83_41", 1289, 346, 1342, 411),
    ("train", "Img_83_59", 1198, 858, 1265, 941),
    ("train", "Img_83_59", 1206, 984, 1267, 1069),
    ("val", "Img_115_87", 98, 322, 139, 385),
    ("val", "Img_115_87", 1359, 768, 1400, 821),
    ("val", "Img_115_87", 1390, 779, 1463, 856),
    ("val", "Img_115_87", 1247, 899, 1360, 994),
    ("val", "Img_115_87", 386, 941, 455, 1040),
    ("val", "Img_115_89", 1172, 0, 1215, 54),
    ("val", "Img_115_89", 1236, 0, 1303, 36),
    ("val", "Img_115_89", 984, 4, 1055, 97),
    ("val", "Img_115_89", 1097, 118, 1202, 183),
    ("val", "Img_115_89", 1046, 135, 1107, 188),
    ("val", "Img_115_89", 1546, 554, 1599, 605),
    ("val", "Img_116_35", 1107, 23, 1184, 66),
]

by_image = {}
for split, stem, x1, y1, x2, y2 in SUSPICIOUS:
    by_image.setdefault((split, stem), []).append((x1, y1, x2, y2))

total_removed = 0
for (split, stem), boxes in by_image.items():
    label_path = DS / "labels" / split / f"{stem}.txt"
    img_path = DS / "images" / split / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    kept = []
    removed = 0
    for line in lines:
        cls, cx, cy, w, h = [float(v) for v in line.split()]
        x1 = int((cx - w/2)*W); y1 = int((cy - h/2)*H)
        x2 = int((cx + w/2)*W); y2 = int((cy + h/2)*H)
        match = any(abs(x1-bx1)<=1 and abs(y1-by1)<=1 and abs(x2-bx2)<=1 and abs(y2-by2)<=1
                    for (bx1,by1,bx2,by2) in boxes)
        if match:
            removed += 1
        else:
            kept.append(line)
    label_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    total_removed += removed
    print(f"{split}/{stem}: removed {removed}/{len(boxes)} expected, {len(kept)} boxes remain")

print(f"\nTotal removed: {total_removed} boxes across {len(by_image)} images")
