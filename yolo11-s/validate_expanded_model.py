from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

RUN_DIR = Path(r"F:\Livo\Data - 2026\Rbc\rbc_yolo\yolo11s_1024_singlecls_expanded-2")
BEST = RUN_DIR / "weights" / "best.pt"
DATASET_YAML = r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded\dataset.yaml"
TEST_IMAGES_DIR = Path(r"F:\Livo\Data - 2026\Rbc\test_folder")
OUT_DIR = Path(r"F:\Livo\Data - 2026\Rbc\rbc_yolo11s_expanded_overlays")
CONF = 0.05
IOU = 0.7
IMGSZ = 1024
BOX_COLOR = (0, 160, 0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(str(BEST))

print("=== Official .val() on the full val set ===")
metrics = model.val(data=DATASET_YAML, workers=0)
print("mAP50   :", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall   :", metrics.box.mr)

print("\n=== Ground-truth recall check at deployed conf=0.05 (val set) ===")
VAL_IMG_DIR = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded\images\val")
VAL_LBL_DIR = Path(r"F:\Livo\Data - 2026\Rbc\yolo_dataset_expanded\labels\val")
val_images = sorted(VAL_IMG_DIR.glob("*.jpg"))
total_gt, total_pred, total_images = 0, 0, 0
for img_path in val_images:
    lbl_path = VAL_LBL_DIR / f"{img_path.stem}.txt"
    gt_count = sum(1 for line in lbl_path.read_text().splitlines() if line.strip())
    img = cv2.imread(str(img_path))
    results = model.predict(img, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    pred_count = len(results.boxes)
    total_gt += gt_count
    total_pred += pred_count
    total_images += 1
print(f"images={total_images}  ground_truth_cells={total_gt}  predicted_boxes={total_pred}  "
      f"recall~={total_pred/total_gt*100:.2f}%  missing~={total_gt-total_pred} ({(total_gt-total_pred)/total_gt*100:.2f}%)")

print(f"\n=== Saving box overlays on {TEST_IMAGES_DIR} ===")
test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
for img_path in test_images:
    img = cv2.imread(str(img_path))
    results = model.predict(img, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    vis = img.copy()
    for b in results.boxes:
        x1, y1, x2, y2 = [round(float(v)) for v in b.xyxy[0]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 1)
    out_path = OUT_DIR / f"{img_path.stem}_boxes.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"  {img_path.name}: {len(results.boxes)} boxes -> {out_path.name}")

print(f"\nSaved to {OUT_DIR}")
