#!/usr/bin/env python3
"""
Пост-обработка BEV маски: сглаживание дорог и зданий.

- Gaussian blur + threshold per class — убирает зубчатые края
- convex hull для зданий (опционально)
- approxPolyDP для сглаживания контуров дорог
- Удаление мелких фрагментов

Usage:
    python smooth_bev_mask.py \
        --input bev_color_mask_dense.png \
        --config input_format_real.yaml \
        --out bev_smoothed.png \
        --road-blur 7 \
        --road-epsilon 5.0
"""

import argparse
import os

import cv2
import numpy as np
import yaml


def load_class_colors(config_path):
    """Загружает маппинг имя → BGR цвет из конфига."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    classes = {}
    for cls in config.get('classes', []):
        if 'mask_color' in cls:
            classes[cls['name']] = {
                'color': np.array(cls['mask_color'], dtype=np.uint8),
                'min_area': cls.get('min_area', config.get('defaults', {}).get('min_area', 100)),
            }
    return classes


def extract_class_mask(image, color, tolerance=0):
    """Извлекает бинарную маску класса по цвету."""
    lower = np.maximum(0, color.astype(np.int16) - tolerance).astype(np.uint8)
    upper = np.minimum(255, color.astype(np.int16) + tolerance).astype(np.uint8)
    return cv2.inRange(image, lower, upper)


def smooth_mask_gaussian(binary_mask, blur_size, threshold=128):
    """Сглаживает маску через Gaussian blur + threshold."""
    if blur_size < 3:
        return binary_mask
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 0)
    _, smoothed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return smoothed


def smooth_contours(binary_mask, epsilon, min_area):
    """Сглаживает контуры через approxPolyDP и удаляет мелкие фрагменты."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.fillPoly(out, [approx], 255)
    return out


def fill_holes_in_mask(binary_mask, max_hole_area=500):
    """Заливает дырки внутри маски (по hierarchy)."""
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return binary_mask
    result = binary_mask.copy()
    for idx, cnt in enumerate(contours):
        parent = hierarchy[0][idx][3]
        if parent == -1:
            continue
        area = cv2.contourArea(cnt)
        if area <= max_hole_area:
            cv2.drawContours(result, [cnt], -1, 255, -1)
    return result


def process_road(binary_mask, blur_size, epsilon, min_area, hole_area):
    """Полный pipeline для дорог: blur → fill holes → smooth contours."""
    # 1. Морф-закрытие для объединения близких фрагментов
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Gaussian blur для сглаживания зубцов
    mask = smooth_mask_gaussian(mask, blur_size)

    # 3. Заливка дырок
    mask = fill_holes_in_mask(mask, hole_area)

    # 4. Сглаживание контуров (Douglas-Peucker)
    mask = smooth_contours(mask, epsilon, min_area)

    return mask


def process_building(binary_mask, blur_size, epsilon, min_area):
    """Pipeline для зданий: blur → smooth contours."""
    mask = smooth_mask_gaussian(binary_mask, blur_size)
    mask = smooth_contours(mask, epsilon, min_area)
    return mask


def process_generic(binary_mask, blur_size, epsilon, min_area):
    """Pipeline для прочих классов: лёгкое сглаживание."""
    mask = smooth_mask_gaussian(binary_mask, blur_size)
    mask = smooth_contours(mask, epsilon, min_area)
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Сглаживание BEV маски: дороги, здания, прочие классы"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Входная BEV маска (PNG)")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML конфиг с цветами классов")
    parser.add_argument("--out", type=str, required=True,
                        help="Выходная сглаженная маска (PNG)")

    parser.add_argument("--road-blur", type=int, default=9,
                        help="Gaussian blur kernel для дорог (default: 9)")
    parser.add_argument("--road-epsilon", type=float, default=5.0,
                        help="approxPolyDP epsilon для дорог (default: 5.0)")
    parser.add_argument("--road-min-area", type=int, default=200,
                        help="Мин. площадь дорожного полигона (default: 200)")
    parser.add_argument("--road-hole-area", type=int, default=500,
                        help="Макс. площадь дырки для заливки в дорогах (default: 500)")

    parser.add_argument("--building-blur", type=int, default=5,
                        help="Gaussian blur для зданий (default: 5)")
    parser.add_argument("--building-epsilon", type=float, default=3.0,
                        help="approxPolyDP epsilon для зданий (default: 3.0)")

    parser.add_argument("--generic-blur", type=int, default=5,
                        help="Gaussian blur для прочих классов (default: 5)")
    parser.add_argument("--generic-epsilon", type=float, default=3.0,
                        help="approxPolyDP epsilon для прочих классов (default: 3.0)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Маска не найдена: {args.input}")

    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Не удалось загрузить: {args.input}")

    classes = load_class_colors(args.config)
    h, w = image.shape[:2]
    result = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Определяем какие классы — дороги
    road_names = {"road_surface", "traffic_road", "parking", "ground", "footpath"}
    building_names = {"building"}

    # Порядок отрисовки: сначала большие (дороги), потом мелкие поверх
    draw_order = []
    for name, info in classes.items():
        if name.lower() in road_names:
            priority = 0
        elif name.lower() in building_names:
            priority = 2
        else:
            priority = 1
        draw_order.append((priority, name, info))
    draw_order.sort(key=lambda x: x[0])

    for priority, name, info in draw_order:
        color = info['color']
        min_area = info['min_area']

        binary = extract_class_mask(image, color)
        n_pixels = int((binary > 0).sum())
        if n_pixels == 0:
            continue

        if name.lower() in road_names:
            smoothed = process_road(
                binary, args.road_blur, args.road_epsilon,
                args.road_min_area, args.road_hole_area
            )
            label = "road"
        elif name.lower() in building_names:
            smoothed = process_building(
                binary, args.building_blur, args.building_epsilon, min_area
            )
            label = "building"
        else:
            smoothed = process_generic(
                binary, args.generic_blur, args.generic_epsilon, min_area
            )
            label = "generic"

        n_after = int((smoothed > 0).sum())
        print(f"  {name} ({label}): {n_pixels:,} → {n_after:,} px")

        # Рисуем сглаженную маску оригинальным цветом
        result[smoothed > 0] = color

    cv2.imwrite(args.out, result)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
