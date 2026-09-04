"""
Prototype: per-cell classification (size, color, shape) using pure classical
CV, applied to isolated cells found by classical_isolated_cell_contours.

Thresholds are PERCENTILE-based (relative to this image's own cell
population), since there's no externally calibrated pixel-to-micron scale
or clinically-validated absolute cutoffs available yet -- this is a
prototype to demonstrate the method, not a clinically calibrated tool.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype_classical_only import classical_isolated_cell_contours, _foreground_mask

WBC_HUE_MIN, WBC_HUE_MAX = 120, 155


def _smooth_for_shape_analysis(pts, window=9):
    """Light smoothing used ONLY for shape-descriptor computation (circularity,
    radial signature) -- the raw findContours output retains pixel-level
    staircase noise that inflates irregularity scores even for genuinely
    round cells. This does NOT touch the stored/output contour points, only
    the copy used to measure shape."""
    n = len(pts)
    if n < window * 2:
        return pts
    pad = window // 2
    ext = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0)
    kernel = np.ones(window, dtype=np.float32) / window
    xs = np.convolve(ext[:, 0], kernel, mode='valid')
    ys = np.convolve(ext[:, 1], kernel, mode='valid')
    return np.stack([xs, ys], axis=1).astype(np.float32)


def radial_signature(pts, n_bins=72):
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.hypot(dx, dy)
    order = np.argsort(angles)
    angles, radii = angles[order], radii[order]
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if m.any():
            binned[i] = radii[m].mean()
    valid = ~np.isnan(binned)
    if valid.sum() < n_bins:
        binned[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), binned[valid])
    return binned


def cell_features(pts, hsv, img_bgr):
    smoothed_pts = _smooth_for_shape_analysis(pts)
    c = smoothed_pts.reshape(-1, 1, 2).astype(np.float32)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri ** 2) if peri > 0 else 0

    sig = radial_signature(smoothed_pts)
    radial_cv = sig.std() / sig.mean() if sig.mean() > 0 else 0

    bx, by, bw, bh = cv2.boundingRect(c)
    local_mask = np.zeros((bh, bw), np.uint8)
    cv2.drawContours(local_mask, [c.astype(np.int32) - [bx, by]], -1, 255, -1)
    region = local_mask > 0
    hsv_crop = hsv[by:by + bh, bx:bx + bw]
    sat = hsv_crop[:, :, 1][region]
    mean_sat = float(sat.mean()) if sat.size else 0.0

    return {
        "area": area,
        "circularity": circularity,
        "radial_cv": radial_cv,
        "mean_sat": mean_sat,
        "bbox": (bx, by, bw, bh),
    }


def classify_all(cells_features):
    areas = np.array([f["area"] for f in cells_features])
    sats = np.array([f["mean_sat"] for f in cells_features])

    area_lo, area_hi = np.percentile(areas, [20, 80])
    sat_lo, sat_hi = np.percentile(sats, [20, 80])

    for f in cells_features:
        # SIZE
        if f["area"] < area_lo:
            f["size_label"] = "Microcyte"
        elif f["area"] > area_hi:
            f["size_label"] = "Macrocyte"
        else:
            f["size_label"] = "Normocyte"

        # COLOR (mean saturation as proxy for chromicity)
        if f["mean_sat"] < sat_lo:
            f["color_label"] = "Hypochromatic"
        elif f["mean_sat"] > sat_hi:
            f["color_label"] = "Hyperchromatic"
        else:
            f["color_label"] = "Normal"

        # SHAPE (combined circularity + radial-signature irregularity)
        if f["circularity"] >= 0.85 and f["radial_cv"] < 0.05:
            f["shape_label"] = "Normal"
        else:
            f["shape_label"] = "Irregular"

    return cells_features


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Livo\Data - 2026\Rbc\others\7\Img_7_13.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("could not read", img_path)
        return

    contours = classical_isolated_cell_contours(img, min_area=3500, max_area=7500, min_circularity=0.75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = [cell_features(pts, hsv, img) for pts in contours]
    features = classify_all(features)

    print(f"{len(features)} cells classified\n")
    print(f"{'#':>3s} {'area':>6s} {'circ':>5s} {'rcv':>6s} {'sat':>5s}   {'SIZE':<11s} {'COLOR':<15s} {'SHAPE':<10s}")
    for i, f in enumerate(features):
        print(f"{i:3d} {f['area']:6.0f} {f['circularity']:5.2f} {f['radial_cv']:6.3f} {f['mean_sat']:5.1f}   "
              f"{f['size_label']:<11s} {f['color_label']:<15s} {f['shape_label']:<10s}")

    from collections import Counter
    print("\nSize distribution:", dict(Counter(f["size_label"] for f in features)))
    print("Color distribution:", dict(Counter(f["color_label"] for f in features)))
    print("Shape distribution:", dict(Counter(f["shape_label"] for f in features)))

    # overlay: color-code by shape label, annotate size/color as short tags
    vis = img.copy()
    shape_colors = {"Normal": (0, 255, 0), "Irregular": (0, 0, 255)}
    size_tag = {"Microcyte": "mi", "Normocyte": "no", "Macrocyte": "ma"}
    color_tag = {"Hypochromatic": "hy", "Normal": "n", "Hyperchromatic": "hc"}
    for pts, f in zip(contours, features):
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        col = shape_colors[f["shape_label"]]
        cv2.polylines(vis, [poly], True, col, 1, cv2.LINE_AA)
        bx, by, bw, bh = f["bbox"]
        label = f"{size_tag[f['size_label']]}/{color_tag[f['color_label']]}"
        cv2.putText(vis, label, (bx, by - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1, cv2.LINE_AA)

    out_path = str(Path(__file__).parent / "classify_test_overlay.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\nsaved overlay -> {out_path}")


if __name__ == "__main__":
    main()
