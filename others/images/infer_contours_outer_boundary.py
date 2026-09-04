"""
RBC bounding-box detection + contour extraction, standalone (no notebook needed).

Pipeline: YOLO (box-only, single_cls model) -> one whole-image Otsu threshold
-> connected components. A box only gets a contour if its own connected
foreground blob contains no other box -- i.e. it's a genuinely isolated,
single RBC. Touching/overlapping cells (doubles, triples, any larger
cluster) share one blob with another box and get no contour at all.

This is a variant of infer_contours.py that drops the per-cell watershed
split. The original script is left untouched.

Run:
    "F:\\envs\\rbc\\python.exe" "F:\\Livo\\Data - 2026\\Rbc\\infer_contours_outer_boundary.py"
"""
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BEST_PT     = Path('F:/Livo/Data - 2026/Rbc/rbc_yolo/yolo11s_1024_singlecls_expanded-2/weights/best.pt')
IMAGES_DIR  = Path(r'F:\Livo\Data - 2026\1753802296\testdata')
OUT_DIR     = Path('F:/Livo/Data - 2026/Rbc/rbc_contour_output_outer_boundary')
N_IMAGES    = 10          # None = run all images in IMAGES_DIR

IMGSZ       = 1024
CONF_THRES  = 0.10        # 0.05 let through a handful of low-confidence false
                           # positives on WBCs (~0.27% of training labels were
                           # leaked WBC instances); every WBC-flagged detection
                           # measured so far tops out at conf=0.089, while real
                           # RBC detections lost between 0.05 and 0.10 are <1%
IOU_THRES   = 0.7
CROP_PAD    = 25          # px padding when cropping a component's region back
                           # out for cv2.findContours
SMOOTH_WINDOW = 9         # circular moving-average window (px of boundary,
                           # not image px) used to smooth out the pixel
                           # staircase that cv2.findContours produces
SIMPLIFY_EPS  = 0.3       # approxPolyDP epsilon applied AFTER smoothing, to
                           # collapse the smoothed curve's dense point cloud
                           # (~200+ pts/cell from CHAIN_APPROX_NONE) back down
                           # to a compact polygon -- lowered from 1.2 since
                           # that value made cells look visibly faceted
                           # (~16 pts/cell); 0.3 keeps ~2x the points
                           # (~34 avg, up to ~58) for a rounder, more
                           # detailed outline while still cutting the raw
                           # point cloud down substantially
DEDUP_CONTAINMENT_THRESH = 0.5   # a box nested this much inside another box
                           # is treated as a near-duplicate detection of the
                           # SAME physical cell, not a second cell -- without
                           # this, a low-confidence spurious duplicate next to
                           # a real single cell inflates that cell's blob to
                           # "2 boxes" and wrongly excludes it as a cluster
DEDUP_CENTER_FRAC = 0.3    # high containment ALONE isn't enough to call two
                           # boxes duplicates of the same cell -- two
                           # genuinely different, touching cells can also have
                           # one box sitting mostly inside the other's. Only
                           # treat it as a true duplicate if the centers are
                           # also this close (as a fraction of box size);
                           # otherwise both boxes are kept, and the shared-
                           # blob touching check below correctly excludes them
                           # as a real overlapping pair instead of silently
                           # discarding one and leaving the other looking
                           # falsely isolated
MIN_CONTOUR_AREA = 900     # px^2 -- rejects platelet/debris-sized specks
                           # (~10-30px objects); real RBC contours run
                           # ~5000-7000px^2. Exists because a whole-image
                           # Otsu threshold can catastrophically fail on an
                           # unusually pale image (measured: 0.6% foreground
                           # coverage vs the normal ~48%), leaving only a few
                           # high-saturation debris specks as "foreground" --
                           # a nearby box's majority-overlap can then land on
                           # one of those specks instead of the real (but
                           # undetected) cell
WBC_HUE_MIN, WBC_HUE_MAX = 120, 155   # violet/purple WBC-nucleus hue range
WBC_SAT_MIN = 40
WBC_REJECT_FRAC = 0.35     # reject a candidate shape if this much of its OWN
                           # pixels look WBC-colored -- catches the case where
                           # a box's majority-overlap blob turned out to
                           # include real WBC material (the RBC hue filter
                           # above isn't fully exclusive of WBC hue, so a WBC
                           # can pass it and fuse into a nearby cell's blob)
                           # even though the model itself never boxed the WBC
RBC_HUE_MIN     = 130     # OpenCV hue (0-179). A pixel counts as RBC only if
RBC_HUE_MAX_WRAP = 15     # hue >= RBC_HUE_MIN or hue <= RBC_HUE_MAX_WRAP (the
                           # pink/magenta RBC hue wraps around the 179/0
                           # boundary) -- keeps yellow/green debris specks from
                           # bridging two separate cells into one blob
ERODE_KERNEL = np.ones((3, 3), np.uint8)
ERODE_ITERS  = 2          # used only to test whether an already-flagged
                           # "shared blob" (2+ boxes) is a real touching pair
                           # or two close-but-separate cells bridged by a
                           # thin (1-2px) sliver of borderline-saturation
                           # pixels (JPEG blur/antialiasing at a near-gap).
                           # Verified empirically: a real touching pair's
                           # contact survives 4+ erosion iterations as one
                           # component; a false bridge from background bleed
                           # splits cleanly by iteration 2.
BOX_COLOR   = (0, 160, 0)
CONTOUR_COLOR = (0, 160, 0)
ENABLE_SMOOTHING = True   # set False to get the FULL raw cv2.findContours
                           # output (CHAIN_APPROX_NONE, every boundary pixel)
                           # with NEITHER the moving-average smoothing NOR
                           # the approxPolyDP simplification applied -- the
                           # maximum-detail contour, for comparison against
                           # the smoothed+simplified default


_CLOSE_KERNEL = np.ones((3, 3), np.uint8)


def _smooth_closed_contour(pts, window=SMOOTH_WINDOW):
    """Circular moving-average over a closed contour's points. cv2.findContours
    traces the mask pixel-by-pixel, so its raw output is a staircase even for
    a genuinely round cell; averaging each point with its neighbors (wrapping
    around, since the contour is a loop) turns that staircase into a smooth
    curve without changing the cell's actual shape."""
    n = len(pts)
    if n < window * 2:
        return pts
    pad = window // 2
    ext = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0)
    kernel = np.ones(window, dtype=np.float32) / window
    xs = np.convolve(ext[:, 0], kernel, mode='valid')
    ys = np.convolve(ext[:, 1], kernel, mode='valid')
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _foreground_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, fg = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # RBCs stain a pink/magenta hue; small debris/stain-flecks between two
    # otherwise separate cells often land on a very different hue (e.g.
    # yellow-green) and can bridge them into one connected blob even though
    # the cells themselves never touch. Excluding non-RBC hues keeps such
    # debris from ever acting as a bridge; confirmed on real data to remove
    # zero genuine RBC pixels while fixing several false-merge cases.
    hue = hsv[:, :, 0]
    rbc_hue = (hue >= RBC_HUE_MIN) | (hue <= RBC_HUE_MAX_WRAP)
    fg[~rbc_hue] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return fg


def _dedup_boxes(boxes, confidences, containment_thresh=DEDUP_CONTAINMENT_THRESH,
                  center_frac=DEDUP_CENTER_FRAC):
    """Flag a box as a near-duplicate when it's almost entirely nested inside
    another, higher-confidence box AND its center is close to that box's own
    center -- standard NMS (by IoU) can miss the containment case, since a
    small box fully inside a much larger one can have moderate IoU (skewed by
    the size difference) even though it's clearly redundant. The center check
    matters because high containment alone isn't unique to duplicates: two
    genuinely different, touching cells can also have one box sitting mostly
    inside the other's -- discarding that box would silently erase a real
    second cell and leave the container looking falsely isolated instead of
    correctly excluded as a touching pair. Real duplicates of the same cell
    sit almost on top of each other; two different overlapping cells don't.
    Returns a keep-mask the same length as boxes."""
    order = sorted(range(len(boxes)), key=lambda i: -confidences[i])
    keep = [True] * len(boxes)
    for ai in range(len(order)):
        ia = order[ai]
        if not keep[ia]:
            continue
        ax1, ay1, ax2, ay2 = boxes[ia]
        area_a = (ax2 - ax1) * (ay2 - ay1)
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        a_size = max(ax2 - ax1, ay2 - ay1)
        for bi in range(ai + 1, len(order)):
            ib = order[bi]
            if not keep[ib]:
                continue
            bx1, by1, bx2, by2 = boxes[ib]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_b = (bx2 - bx1) * (by2 - by1)
            smaller = min(area_a, area_b)
            if not (smaller and inter / smaller > containment_thresh):
                continue
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            b_size = max(bx2 - bx1, by2 - by1)
            center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            if center_dist < center_frac * max(a_size, b_size):
                keep[ib] = False
    return keep


def _validate_shape(img_bgr, region, ox, oy):
    """Reject a candidate cell shape before it's accepted: too small (a
    platelet/debris speck, not a real RBC) or too WBC-colored (this box's
    majority-overlap blob included real WBC material). `region` is the
    binary mask (0/255) in local crop coords; (ox,oy) offsets it back to
    full-image coords for sampling the original color."""
    area = int(np.count_nonzero(region))
    if area < MIN_CONTOUR_AREA:
        return False
    h, w = region.shape
    crop = img_bgr[oy:oy + h, ox:ox + w]
    if crop.shape[:2] != region.shape:
        return True   # crop got clipped at an image edge -- skip the color check rather than false-reject
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = region > 0
    hue = hsv[:, :, 0][mask]
    sat = hsv[:, :, 1][mask]
    if hue.size == 0:
        return False
    wbc_frac = ((hue >= WBC_HUE_MIN) & (hue <= WBC_HUE_MAX) & (sat >= WBC_SAT_MIN)).mean()
    return wbc_frac < WBC_REJECT_FRAC


def isolated_cell_contours(img_bgr, boxes, confidences):
    """Draw a contour ONLY for boxes that are physically isolated -- their
    own connected foreground blob contains no other (non-duplicate) box. A
    blob shared by 2+ DISTINCT boxes means touching/overlapping cells
    (double, triple, or any larger cluster); those get no contour at all,
    since only single, separate RBCs should get an outline. Near-duplicate
    detections of the same physical cell (see _dedup_boxes) don't count as a
    second cell -- otherwise a real single cell with a spurious low-confidence
    duplicate box next to it would be wrongly excluded as a "cluster". A
    box's own component is found by majority-overlap (not just its center
    pixel, which can land on a pale cell's hollow center and miss the label).

    Before giving up on a shared blob, it's tested with local erosion (see
    ERODE_ITERS) to rule out a false merge: two genuinely separate, close
    cells can get bridged into one blob by a thin sliver of borderline-
    saturation pixels (background bleed/JPEG blur right at their near-gap)
    even though nothing actually touches. A real touching pair's contact is
    much wider and survives the erosion; a false bridge doesn't, so each box
    that ends up in its own erosion-only component gets its true full-size
    contour back (region eroded then dilated by the same amount).

    Returns a list the same length as boxes."""
    H, W = img_bgr.shape[:2]
    fg = _foreground_mask(img_bgr)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    box_label = []
    for x1, y1, x2, y2 in boxes:
        region = labels[y1:y2, x1:x2]
        counts = np.bincount(region.ravel(), minlength=n_labels)
        counts[0] = 0   # background label never wins
        box_label.append(int(counts.argmax()) if counts.max() > 0 else 0)

    # a label shared by 2+ DISTINCT (non-duplicate) boxes is a touching/
    # overlapping cluster -- skip it, UNLESS erosion shows it's actually
    # several genuinely separate cells bridged by a thin false connection
    # (see ERODE_ITERS above)
    keep = _dedup_boxes(boxes, confidences)
    label_counts = Counter(l for l, k in zip(box_label, keep) if l != 0 and k)

    unmerged = {}   # box index -> (region_uint8, offset_x, offset_y)
    for lbl in set(box_label):
        if lbl == 0 or label_counts[lbl] < 2:
            continue
        member_boxes = [i for i, (l, k) in enumerate(zip(box_label, keep)) if l == lbl and k]
        x, y, w, h, _area = stats[lbl]
        cx1, cy1 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
        cx2, cy2 = min(W, x + w + CROP_PAD), min(H, y + h + CROP_PAD)
        blob = (labels[cy1:cy2, cx1:cx2] == lbl).astype(np.uint8) * 255
        eroded = cv2.erode(blob, ERODE_KERNEL, iterations=ERODE_ITERS)
        n_sub, sub_labels, _sub_stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)

        sub_of_box = {}
        for i in member_boxes:
            bx1, by1, bx2, by2 = boxes[i]
            rx1, ry1 = max(0, bx1 - cx1), max(0, by1 - cy1)
            rx2, ry2 = min(cx2 - cx1, bx2 - cx1), min(cy2 - cy1, by2 - cy1)
            sub_counts = np.bincount(sub_labels[ry1:ry2, rx1:rx2].ravel(), minlength=n_sub)
            sub_counts[0] = 0
            sub_of_box[i] = int(sub_counts.argmax()) if sub_counts.max() > 0 else 0

        sub_label_counts = Counter(sub_of_box.values())
        for i in member_boxes:
            sl = sub_of_box[i]
            if sl != 0 and sub_label_counts[sl] == 1:
                # this box's own eroded component has no other box in it --
                # genuinely isolated once the false bridge is severed
                sub_mask = (sub_labels == sl).astype(np.uint8) * 255
                restored = cv2.dilate(sub_mask, ERODE_KERNEL, iterations=ERODE_ITERS)
                unmerged[i] = (restored, cx1, cy1)

    out = [None] * len(boxes)
    cache = {}
    for i, lbl in enumerate(box_label):
        if i in unmerged:
            region, ox, oy = unmerged[i]
            if not _validate_shape(img_bgr, region, ox, oy):
                continue
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                best = max(contours, key=cv2.contourArea)
                pts = best.reshape(-1, 2).astype(np.float32)
                pts[:, 0] += ox
                pts[:, 1] += oy
                if ENABLE_SMOOTHING:
                    smoothed = _smooth_closed_contour(pts)
                    pts = cv2.approxPolyDP(smoothed.reshape(-1, 1, 2), SIMPLIFY_EPS, True).reshape(-1, 2)
                out[i] = pts
            continue
        if lbl == 0 or label_counts[lbl] != 1:   # not foreground, or a cluster
            continue
        if lbl not in cache:
            x, y, w, h, _area = stats[lbl]
            cx1, cy1 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
            cx2, cy2 = min(W, x + w + CROP_PAD), min(H, y + h + CROP_PAD)
            region = (labels[cy1:cy2, cx1:cx2] == lbl).astype(np.uint8) * 255
            if not _validate_shape(img_bgr, region, cx1, cy1):
                cache[lbl] = None
                continue
            # CHAIN_APPROX_NONE (every boundary pixel, not just corners) so
            # the smoothing step below has enough points to average over
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                cache[lbl] = None
            else:
                best = max(contours, key=cv2.contourArea)
                pts = best.reshape(-1, 2).astype(np.float32)
                pts[:, 0] += cx1   # back to original image coordinates
                pts[:, 1] += cy1
                if ENABLE_SMOOTHING:
                    smoothed = _smooth_closed_contour(pts)
                    pts = cv2.approxPolyDP(smoothed.reshape(-1, 1, 2), SIMPLIFY_EPS, True).reshape(-1, 2)
                cache[lbl] = pts
        out[i] = cache[lbl]
    return out


def detect_and_contour(model, img_bgr):
    """Runs YOLO box detection, then one whole-image connected-components
    pass for all boxes together. Returns (detections, timing_dict)."""
    H, W = img_bgr.shape[:2]

    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMGSZ, verbose=False)
    detect_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    boxes, confidences = [], []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            boxes.append((x1, y1, x2, y2))
            confidences.append(round(float(box.conf[0]), 4))

    contours = isolated_cell_contours(img_bgr, boxes, confidences)

    dets = []
    for det_id, ((x1, y1, x2, y2), conf, pts) in enumerate(zip(boxes, confidences, contours), start=1):
        det = {
            'id':         det_id,
            'bbox':       {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1},
            'confidence': conf,
        }
        if pts is not None:
            det['contour'] = [[round(float(px), 1), round(float(py), 1)] for px, py in pts]
        dets.append(det)
    contour_ms = (time.perf_counter() - t1) * 1000

    timing = {'detect_ms': round(detect_ms, 2), 'contour_ms': round(contour_ms, 2),
              'total_ms': round(detect_ms + contour_ms, 2)}
    return dets, timing


def draw_boxes(img_bgr, detections):
    """Bounding boxes ONLY."""
    vis = img_bgr.copy()
    for det in detections:
        b = det['bbox']
        cv2.rectangle(vis, (b['x'], b['y']), (b['x'] + b['width'], b['y'] + b['height']), BOX_COLOR, 1)
    return vis


def draw_contours(img_bgr, detections):
    """Contours ONLY (falls back to nothing drawn for a detection with no contour)."""
    vis = img_bgr.copy()
    for det in detections:
        if 'contour' in det:
            pts = np.array(det['contour'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=CONTOUR_COLOR, thickness=1, lineType=cv2.LINE_AA)
    return vis


def main():
    BOX_OVERLAY_DIR     = OUT_DIR / 'overlays_boxes'
    CONTOUR_OVERLAY_DIR = OUT_DIR / 'overlays_contours'
    JSON_OUT            = OUT_DIR / 'json'
    BOX_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    CONTOUR_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BEST_PT))
    images = sorted(IMAGES_DIR.glob('*.jpg'))
    if N_IMAGES is not None:
        images = images[:N_IMAGES]
    print(f'Model: {BEST_PT}')
    print(f'Running on {len(images)} images -> {OUT_DIR}\n')

    all_timings = []
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [{idx:3d}] SKIP (unreadable): {img_path.name}')
            continue

        dets, timing = detect_and_contour(model, img)
        n_contours = sum(1 for d in dets if 'contour' in d)
        all_timings.append(timing)

        box_vis = draw_boxes(img, dets)
        cv2.imwrite(str(BOX_OVERLAY_DIR / (img_path.stem + '_boxes.jpg')), box_vis)

        contour_vis = draw_contours(img, dets)
        cv2.imwrite(str(CONTOUR_OVERLAY_DIR / (img_path.stem + '_contours.jpg')), contour_vis)

        with open(JSON_OUT / (img_path.stem + '_pred.json'), 'w') as f:
            json.dump({'image_id': img_path.name, 'detections': dets, 'timing': timing}, f, indent=2)

        print(f"  [{idx:3d}/{len(images)}] {img_path.name:<22s}  {len(dets):3d} boxes  "
              f"{n_contours:3d} contours  detect={timing['detect_ms']:6.1f}ms  "
              f"contour={timing['contour_ms']:6.1f}ms  total={timing['total_ms']:6.1f}ms")

    print(f'\nBox overlays     -> {BOX_OVERLAY_DIR}')
    print(f'Contour overlays -> {CONTOUR_OVERLAY_DIR}')
    print(f'JSONs            -> {JSON_OUT}')

    if all_timings:
        n = len(all_timings)
        avg_detect  = sum(t['detect_ms']  for t in all_timings) / n
        avg_contour = sum(t['contour_ms'] for t in all_timings) / n
        avg_total   = sum(t['total_ms']   for t in all_timings) / n
        print(f'\nAverage over {n} images: detect={avg_detect:6.1f}ms  '
              f'contour={avg_contour:6.1f}ms  total={avg_total:6.1f}ms')


if __name__ == '__main__':
    main()
