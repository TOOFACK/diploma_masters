#!/usr/bin/env python3
"""
Скрипт для векторизации масок сегментации.

Извлекает объекты заданных классов по цвету из YAML конфига, аппроксимирует их примитивами
(полигоны через Douglas-Peucker, прямоугольники, эллипсы) и сохраняет в JSON и PNG форматах.

Не использует shapely для полигонов — только OpenCV.
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
from road_processing import RoadMaskProcessor


def load_config(config_path: str) -> Dict:
    """Загружает конфигурацию классов из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Не удалось загрузить конфиг из {config_path}")

    if 'classes' not in config:
        raise ValueError("Конфиг должен содержать ключ 'classes'")

    # Нормализуем структуру: если classes - словарь, преобразуем в список
    if isinstance(config['classes'], dict):
        if 'name' in config['classes']:
            config['classes'] = [config['classes'].copy()]
        else:
            classes_list = []
            for key, value in config['classes'].items():
                if isinstance(value, dict):
                    class_dict = value.copy()
                    if 'name' not in class_dict:
                        class_dict['name'] = key
                    classes_list.append(class_dict)
                else:
                    classes_list.append({'name': key})
            config['classes'] = classes_list
    elif not isinstance(config['classes'], list):
        raise ValueError("'classes' должен быть списком или словарём")

    return config


def get_class_param(class_config: Dict, defaults: Dict, key: str, fallback):
    """Возвращает параметр класса с учётом defaults и fallback."""
    return class_config.get(key, defaults.get(key, fallback))


def extract_class_mask(mask: np.ndarray, mask_color: List[int],
                       tolerance: int, color_space: str = 'BGR') -> np.ndarray:
    """Извлекает бинарную маску для заданного цвета с допуском."""
    if color_space == 'RGB':
        target_color = np.array([mask_color[2], mask_color[1], mask_color[0]], dtype=np.uint8)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR) if len(mask.shape) == 3 else mask
    else:
        target_color = np.array(mask_color, dtype=np.uint8)
        mask_bgr = mask

    lower = np.maximum(0, target_color - tolerance).astype(np.uint8)
    upper = np.minimum(255, target_color + tolerance).astype(np.uint8)
    binary = cv2.inRange(mask_bgr, lower, upper)
    return binary


def clean_mask(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Применяет морфологическую очистку (opening + closing)."""
    if kernel_size < 2:
        return binary_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def find_objects(binary_mask: np.ndarray, min_area: int) -> List[Tuple[np.ndarray, float]]:
    """Находит объекты (контуры) в бинарной маске и фильтрует по минимальной площади."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            objects.append((contour, area))
    return objects


def fit_primitive(contour: np.ndarray, area: float, primitive_type: str,
                  epsilon: float) -> Dict[str, Any]:
    """
    Аппроксимирует контур примитивом заданного типа.
    polygon — cv2.approxPolyDP (Douglas-Peucker), без shapely.
    """
    geometry = {}

    if primitive_type == 'polygon':
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            geometry['points'] = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        else:
            geometry['points'] = [[int(p[0]), int(p[1])] for p in points]

    elif primitive_type == 'rectangle':
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)
        center = rect[0]
        size = rect[1]
        angle = rect[2]

        geometry['box_points'] = [[int(p[0]), int(p[1])] for p in box_points]
        geometry['center'] = [float(center[0]), float(center[1])]
        geometry['size'] = [float(size[0]), float(size[1])]
        geometry['angle_deg'] = float(angle)

    elif primitive_type == 'ellipse':
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center = ellipse[0]
            axes = ellipse[1]
            angle = ellipse[2]

            geometry['type'] = 'ellipse'
            geometry['center'] = [float(center[0]), float(center[1])]
            geometry['axes'] = [float(axes[0]), float(axes[1])]
            geometry['angle_deg'] = float(angle)
        else:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            geometry['type'] = 'circle'
            geometry['center'] = [float(x), float(y)]
            geometry['radius'] = float(radius)

    return geometry


def compute_anchor(contour: np.ndarray, geometry: Dict[str, Any],
                   primitive_type: str) -> Tuple[float, float]:
    """Вычисляет точку привязки (anchor) для объекта."""
    if primitive_type in ('rectangle', 'ellipse') and 'center' in geometry:
        return tuple(geometry['center'])
    # Для polygon: момент масс контура
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return (cx, cy)
    # Fallback: центр bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w / 2.0, y + h / 2.0)


def render_preview(mask_shape: Tuple[int, int], layers: List[Dict],
                   config: Dict, draw_labels: bool = False) -> np.ndarray:
    """Отрисовывает превью с примитивами на белом фоне."""
    height, width = mask_shape
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    render_configs = {}
    for cls in config['classes']:
        render_configs[cls['name']] = cls.get('render', {})

    for layer in layers:
        class_name = layer['class']
        primitive_type = layer['primitive']
        render_cfg = render_configs.get(class_name, {})

        fill_color = render_cfg.get('fill_color', [128, 128, 128])
        stroke_color = render_cfg.get('stroke_color', [0, 0, 0])
        stroke_thickness = render_cfg.get('stroke_thickness', 2)
        fill = render_cfg.get('fill', True)
        label_color = render_cfg.get('label_color', [0, 0, 0])

        fill_bgr = tuple(fill_color[::-1]) if len(fill_color) == 3 else tuple(fill_color)
        stroke_bgr = tuple(stroke_color[::-1]) if len(stroke_color) == 3 else tuple(stroke_color)
        label_bgr = tuple(label_color[::-1]) if len(label_color) == 3 else tuple(label_color)

        for obj in layer['objects']:
            geometry = obj['geometry']

            if primitive_type == 'polygon' and 'points' in geometry:
                points = np.array(geometry['points'], dtype=np.int32)
                if fill:
                    cv2.fillPoly(canvas, [points], fill_bgr)
                cv2.polylines(canvas, [points], True, stroke_bgr, stroke_thickness)

            elif primitive_type == 'rectangle' or 'box_points' in geometry:
                box_points = np.array(geometry['box_points'], dtype=np.int32)
                if fill:
                    cv2.fillPoly(canvas, [box_points], fill_bgr)
                cv2.polylines(canvas, [box_points], True, stroke_bgr, stroke_thickness)

            elif primitive_type == 'ellipse':
                if geometry.get('type') == 'ellipse':
                    center = tuple(int(x) for x in geometry['center'])
                    axes = tuple(int(x // 2) for x in geometry['axes'])
                    angle = int(geometry['angle_deg'])
                    if fill:
                        cv2.ellipse(canvas, center, axes, angle, 0, 360, fill_bgr, -1)
                    cv2.ellipse(canvas, center, axes, angle, 0, 360, stroke_bgr, stroke_thickness)
                elif 'radius' in geometry:
                    center = tuple(int(x) for x in geometry['center'])
                    radius = int(geometry['radius'])
                    if fill:
                        cv2.circle(canvas, center, radius, fill_bgr, -1)
                    cv2.circle(canvas, center, radius, stroke_bgr, stroke_thickness)

            if draw_labels:
                anchor = obj['anchor']
                anchor_int = (int(anchor[0]), int(anchor[1]))
                cv2.putText(canvas, class_name, anchor_int,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_bgr, 1, cv2.LINE_AA)

    return canvas


def save_json(output_path: str, mask_path: str, config_path: str,
              mask_shape: Tuple[int, int], layers: List[Dict],
              tolerance: int, defaults: Dict):
    """Сохраняет результаты в JSON файл."""
    output_data = {
        'metadata': {
            'mask_path': str(mask_path),
            'config_path': str(config_path),
            'tolerance': tolerance,
            'defaults': defaults,
        },
        'canvas': {
            'width': int(mask_shape[1]),
            'height': int(mask_shape[0])
        },
        'layers': layers
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_class(mask: np.ndarray, class_config: Dict, defaults: Dict,
                  tolerance: int, color_space: str,
                  debug_dir: Optional[str] = None) -> Dict:
    """Обрабатывает один класс: извлекает объекты и аппроксимирует примитивами."""
    class_name = class_config['name']
    mask_color = class_config['mask_color']

    # Per-class параметры с fallback на defaults
    min_area = int(get_class_param(class_config, defaults, 'min_area', 100))
    epsilon = float(get_class_param(class_config, defaults, 'epsilon', 2.0))
    kernel = int(get_class_param(class_config, defaults, 'kernel', 3))
    tol = int(get_class_param(class_config, defaults, 'tolerance', tolerance))
    primitive_type = get_class_param(class_config, defaults, 'primitive', 'polygon')

    road_params = class_config.get('road_processing', {})

    # 1. Extract binary mask
    binary = extract_class_mask(mask, mask_color, tol, color_space)
    if debug_dir:
        cv2.imwrite(str(Path(debug_dir) / f"{class_name}_binary.png"), binary)

    # 2. Morphological cleanup
    cleaned = clean_mask(binary, kernel)

    # 3. Road processing if enabled
    if road_params.get('enabled', False):
        cleaned = RoadMaskProcessor(road_params).preprocess(
            cleaned, debug_dir=debug_dir, debug_prefix=class_name
        )
    if debug_dir:
        cv2.imwrite(str(Path(debug_dir) / f"{class_name}_cleaned.png"), cleaned)

    # 4. Find contours
    objects_data = find_objects(cleaned, min_area)

    layer = {
        'class': class_name,
        'class_id': class_config.get('class_id', 0),
        'primitive': primitive_type,
        'objects': []
    }

    debug_canvas = None
    if debug_dir:
        debug_canvas = np.zeros_like(cleaned)

    # 5. Fit primitives per contour
    for idx, (contour, area) in enumerate(objects_data):
        geometry = fit_primitive(contour, area, primitive_type, epsilon)
        anchor = compute_anchor(contour, geometry, primitive_type)

        obj = {
            'id': f"{class_name}_{idx + 1:04d}",
            'geometry': geometry,
            'anchor': [float(anchor[0]), float(anchor[1])],
            'area_px': float(area)
        }
        layer['objects'].append(obj)

        if debug_dir:
            cv2.drawContours(debug_canvas, [contour], -1, 255, 2)

    if debug_dir and debug_canvas is not None:
        cv2.imwrite(str(Path(debug_dir) / f"{class_name}_contours.png"), debug_canvas)

    return layer


def main():
    parser = argparse.ArgumentParser(
        description='Векторизация масок сегментации с аппроксимацией примитивами'
    )

    parser.add_argument('--mask', type=str, required=True,
                        help='Путь к маске сегментации (RGB/BGR изображение)')
    parser.add_argument('--config', type=str, required=True,
                        help='Путь к конфигурационному YAML файлу')
    parser.add_argument('--out_json', type=str, required=True,
                        help='Путь к выходному JSON файлу')
    parser.add_argument('--out_png', type=str, required=True,
                        help='Путь к выходному PNG файлу')

    parser.add_argument('--tolerance', type=int, default=0,
                        help='Допуск по цвету для каждого канала (по умолчанию: 0)')
    parser.add_argument('--min_area', type=int, default=None,
                        help='Глобальный override минимальной площади объекта')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Глобальный override параметра approxPolyDP')
    parser.add_argument('--kernel', type=int, default=None,
                        help='Глобальный override размера морфологического ядра')
    parser.add_argument('--label', action='store_true',
                        help='Рисовать подписи классов на превью')
    parser.add_argument('--color_space', type=str, default='BGR', choices=['BGR', 'RGB'],
                        help='Цветовое пространство маски (по умолчанию: BGR)')
    parser.add_argument('--debug_dir', type=str, default=None,
                        help='Директория для сохранения отладочных изображений')

    args = parser.parse_args()

    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Маска не найдена: {args.mask}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Конфиг не найден: {args.config}")

    os.makedirs(os.path.dirname(args.out_json) if os.path.dirname(args.out_json) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png) if os.path.dirname(args.out_png) else '.', exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    config = load_config(args.config)

    # Defaults из конфига + CLI overrides
    defaults = config.get('defaults', {})
    if args.min_area is not None:
        defaults['min_area'] = args.min_area
    if args.epsilon is not None:
        defaults['epsilon'] = args.epsilon
    if args.kernel is not None:
        defaults['kernel'] = args.kernel

    mask = cv2.imread(args.mask)
    if mask is None:
        raise ValueError(f"Не удалось загрузить маску: {args.mask}")

    if args.color_space == 'RGB':
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask_shape = mask.shape[:2]

    layers = []
    stats = {}

    for class_config in config['classes']:
        class_name = class_config['name']
        if 'mask_color' not in class_config:
            print(f"Пропуск класса {class_name}: нет mask_color")
            continue

        print(f"Обработка класса: {class_name}")

        layer = process_class(
            mask, class_config, defaults,
            args.tolerance, args.color_space, args.debug_dir
        )

        layers.append(layer)
        num_objects = len(layer['objects'])
        stats[class_name] = num_objects
        print(f"  Найдено объектов: {num_objects}")

    print("Отрисовка превью...")
    preview = render_preview(mask_shape, layers, config, args.label)
    cv2.imwrite(args.out_png, preview)

    print("Сохранение JSON...")
    save_json(args.out_json, args.mask, args.config, mask_shape, layers,
              args.tolerance, defaults)

    print("\n=== Статистика ===")
    total_objects = 0
    for class_name, count in stats.items():
        print(f"{class_name}: {count} объектов")
        total_objects += count
    print(f"Всего объектов: {total_objects}")
    print(f"\nРезультаты сохранены:")
    print(f"  JSON: {args.out_json}")
    print(f"  PNG:  {args.out_png}")


if __name__ == '__main__':
    main()
