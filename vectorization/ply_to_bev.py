#!/usr/bin/env python3
"""
Проецирует PLY с классовыми цветами в BEV (bird's eye view) маску.

Берёт облако точек (coord + pred labels), проецирует на XY-плоскость,
для каждого пикселя определяет класс по majority vote и раскрашивает.

Поддерживает морфологическое закрытие для заполнения щелей в разреженных сканах.

Usage:
    python ply_to_bev.py \
        --ply pred.ply \
        --pred pred.npy \
        --out bev_mask.png \
        --resolution 0.2 \
        --close-kernel 5
"""

import argparse
import os
from collections import Counter

import cv2
import numpy as np
import open3d as o3d


# SensatUrban 13-class colors (BGR for OpenCV)
DEFAULT_CLASS_COLORS_BGR = np.array([
    [128, 128, 128],  # 0  Ground
    [0, 128, 0],      # 1  Vegetation
    [0, 0, 255],      # 2  Building
    [0, 165, 255],    # 3  Wall
    [128, 0, 128],    # 4  Bridge
    [255, 255, 0],    # 5  Parking
    [19, 69, 139],    # 6  Rail
    [64, 64, 64],     # 7  Traffic Road
    [255, 255, 0],    # 8  Street Furniture
    [255, 0, 0],      # 9  Car
    [203, 192, 255],  # 10 Footpath
    [50, 205, 50],    # 11 Bike
    [128, 0, 0],      # 12 Water
], dtype=np.uint8)

CLASS_NAMES = [
    "Ground", "Vegetation", "Building", "Wall", "Bridge", "Parking",
    "Rail", "Traffic Road", "Street Furniture", "Car", "Footpath",
    "Bike", "Water",
]


def project_to_bev(coord, pred, resolution, class_colors):
    """Проецирует точки на XY-плоскость с majority vote per pixel.

    Returns:
        bev: (H, W, 3) uint8 BGR image
        label_map: (H, W) int8, -1 = background
        metadata: dict with projection params for inverse mapping
    """
    x_min, y_min = coord[:, 0].min(), coord[:, 1].min()
    x_max, y_max = coord[:, 0].max(), coord[:, 1].max()
    W = int(np.ceil((x_max - x_min) / resolution))
    H = int(np.ceil((y_max - y_min) / resolution))

    valid = pred >= 0
    coord_v = coord[valid]
    pred_v = pred[valid]

    px = np.clip(((coord_v[:, 0] - x_min) / resolution).astype(np.int32), 0, W - 1)
    py = np.clip(((coord_v[:, 1] - y_min) / resolution).astype(np.int32), 0, H - 1)

    # Majority vote per pixel
    pixel_classes = {}
    for i in range(len(px)):
        key = (py[i], px[i])
        if key not in pixel_classes:
            pixel_classes[key] = []
        pixel_classes[key].append(pred_v[i])

    bev = np.ones((H, W, 3), dtype=np.uint8) * 255
    label_map = np.full((H, W), -1, dtype=np.int8)

    for (r, c), classes in pixel_classes.items():
        counter = Counter(classes)
        majority = counter.most_common(1)[0][0]
        bev[r, c] = class_colors[majority]
        label_map[r, c] = majority

    # Flip Y so north is up
    bev = np.flipud(bev)
    label_map = np.flipud(label_map)

    metadata = {
        "x_min": float(x_min), "y_min": float(y_min),
        "x_max": float(x_max), "y_max": float(y_max),
        "width": W, "height": H,
        "resolution": resolution,
    }

    return bev, label_map, metadata


def close_gaps(bev, label_map, class_colors, kernel_size):
    """Морфологическое закрытие per-class для заполнения щелей."""
    if kernel_size < 2:
        return bev

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bev_filled = bev.copy()
    num_classes = len(class_colors)

    for cls_id in range(num_classes):
        cls_mask = (label_map == cls_id).astype(np.uint8) * 255
        if cls_mask.sum() == 0:
            continue
        closed = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)
        new_pixels = (closed > 0) & (label_map < 0)
        bev_filled[new_pixels] = class_colors[cls_id]

    return bev_filled


def merge_classes(class_colors, merge_map):
    """Применяет слияние классов: merge_map = {target_id: [source_ids]}.

    Example: merge_map = {0: [5, 7]} — Parking и Traffic Road → цвет Ground.
    """
    colors = class_colors.copy()
    for target, sources in merge_map.items():
        for src in sources:
            colors[src] = colors[target]
    return colors


def main():
    parser = argparse.ArgumentParser(
        description="Проекция PLY с предсказаниями классов в BEV маску"
    )
    parser.add_argument("--ply", type=str, required=True,
                        help="Путь к PLY файлу с координатами")
    parser.add_argument("--pred", type=str, required=True,
                        help="Путь к .npy файлу с предсказаниями классов")
    parser.add_argument("--out", type=str, required=True,
                        help="Путь к выходному PNG файлу")
    parser.add_argument("--resolution", type=float, default=0.2,
                        help="Разрешение BEV в м/пиксель (default: 0.2)")
    parser.add_argument("--close-kernel", type=int, default=5,
                        help="Размер ядра морф. закрытия для заполнения щелей (0 = off, default: 5)")
    parser.add_argument("--merge-roads", action="store_true",
                        help="Объединить Ground(0), Parking(5), Traffic Road(7) в один цвет (серый)")
    parser.add_argument("--save-labels", action="store_true",
                        help="Сохранить label_map.npy рядом с выходным файлом")

    args = parser.parse_args()

    if not os.path.exists(args.ply):
        raise FileNotFoundError(f"PLY не найден: {args.ply}")
    if not os.path.exists(args.pred):
        raise FileNotFoundError(f"Predictions не найдены: {args.pred}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load
    print(f"Loading PLY: {args.ply}")
    pcd = o3d.io.read_point_cloud(args.ply)
    coord = np.asarray(pcd.points, dtype=np.float32)
    pred = np.load(args.pred)
    print(f"  Points: {len(coord):,}, predictions: {len(pred):,}")

    # Class colors
    class_colors = DEFAULT_CLASS_COLORS_BGR.copy()
    if args.merge_roads:
        class_colors = merge_classes(class_colors, {0: [5, 7]})
        print("  Merged: Parking(5), Traffic Road(7) → gray")

    # Project
    print(f"Projecting to BEV (resolution={args.resolution} m/px)...")
    bev, label_map, meta = project_to_bev(coord, pred, args.resolution, class_colors)
    print(f"  BEV size: {meta['width']}x{meta['height']} px")
    filled = int((label_map >= 0).sum())
    total = meta['width'] * meta['height']
    print(f"  Fill: {filled:,} / {total:,} px ({filled / total * 100:.1f}%)")

    # Close gaps
    if args.close_kernel >= 2:
        print(f"Closing gaps (kernel={args.close_kernel})...")
        bev = close_gaps(bev, label_map, class_colors, args.close_kernel)
        filled2 = int(np.any(bev != 255, axis=2).sum())
        print(f"  Fill after closing: {filled2:,} px ({filled2 / total * 100:.1f}%)")

    # Save
    cv2.imwrite(args.out, bev)
    print(f"Saved: {args.out}")

    if args.save_labels:
        labels_path = os.path.splitext(args.out)[0] + "_labels.npy"
        np.save(labels_path, label_map)
        print(f"Saved: {labels_path}")

    # Stats
    print("\nPixels per class:")
    for i in range(len(CLASS_NAMES)):
        cnt = int((label_map == i).sum())
        if cnt > 0:
            print(f"  {CLASS_NAMES[i]}: {cnt:,} px")


if __name__ == "__main__":
    main()
