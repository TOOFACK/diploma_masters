#!/usr/bin/env python3
"""
Скрипт для визуализации тайла point cloud и его системы координат.

Отображает:
- Точки тайла с цветами
- Систему координат (оси X, Y, Z)
- Статистику тайла

Примеры использования:
    # Визуализация тайла
    python visualize_tile.py --tile tiles/tile_0_0.ply
    
    # Визуализация с указанием файла статистики
    python visualize_tile.py --tile tiles/tile_0_0.ply --stats tiles/tile_0_0_stats.json
    
    # Визуализация без системы координат
    python visualize_tile.py --tile tiles/tile_0_0.ply --no-coords
    
    # С настройкой масштаба осей
    python visualize_tile.py --tile tiles/tile_0_0.ply --coord-scale 5.0
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# Импорт вспомогательных функций
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_ply import read_ply

SENSAT_CLASS_NAMES = {
    0: "Ground",
    1: "Vegetation",
    2: "Building",
    3: "Wall",
    4: "Bridge",
    5: "Parking",
    6: "Rail",
    7: "Traffic Road",
    8: "Street Furniture",
    9: "Car",
    10: "Footpath",
    11: "Bike",
    12: "Water",
}

# RGB палитра SensatUrban (индексы 0..12), плюс fallback для unknown
SENSAT_CLASS_COLORS = {
    0: [255, 248, 220],
    1: [220, 220, 220],
    2: [139, 71, 38],
    3: [238, 197, 145],
    4: [70, 130, 180],
    5: [179, 238, 58],
    6: [110, 139, 61],
    7: [105, 105, 105],
    8: [0, 0, 128],
    9: [205, 92, 92],
    10: [244, 164, 96],
    11: [147, 112, 219],
    12: [255, 228, 225],
}
UNKNOWN_CLASS_COLOR = [255, 20, 147]


def load_tile(ply_path: str, stats_path: str = None) -> tuple:
    """
    Загружает тайл и его статистику.
    
    Args:
        ply_path: путь к PLY файлу тайла
        stats_path: путь к JSON файлу со статистикой (опционально)
        
    Returns:
        (points, stats) где points - массив точек, stats - словарь статистики
    """
    # Загрузка PLY файла
    ply_data = read_ply(ply_path)
    
    # Извлечение координат
    x = ply_data["x"]
    y = ply_data["y"]
    z = ply_data["z"]
    
    # Извлечение RGB
    if "red" in ply_data.dtype.names:
        r = ply_data["red"]
        g = ply_data["green"]
        b = ply_data["blue"]
    elif "alpha" in ply_data.dtype.names:
        a = ply_data["alpha"]
        r = a.copy()
        g = a.copy()
        b = a.copy()
    else:
        r = np.zeros_like(x, dtype=np.uint8)
        g = np.zeros_like(x, dtype=np.uint8)
        b = np.zeros_like(x, dtype=np.uint8)
    
    # Метки классов (если есть)
    labels = None
    if "class" in ply_data.dtype.names:
        labels = ply_data["class"]
    elif "cla" in ply_data.dtype.names:
        labels = ply_data["cla"]

    # Формирование массива точек [x, y, z, r, g, b]
    points = np.vstack((x, y, z, r, g, b)).T
    
    # Загрузка статистики
    stats = None
    if stats_path is None:
        # Попытка найти файл статистики автоматически
        stats_path = ply_path.replace('.ply', '_stats.json')
    
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    return points, labels, stats


def class_labels_to_colors(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Преобразует массив меток классов в RGB цвета [0,1]."""
    if labels is None:
        return None
    labels_int = labels.astype(np.int32)
    colors = np.zeros((labels_int.shape[0], 3), dtype=np.float32)
    for cls in np.unique(labels_int):
        rgb = SENSAT_CLASS_COLORS.get(int(cls), UNKNOWN_CLASS_COLOR)
        colors[labels_int == cls] = np.asarray(rgb, dtype=np.float32) / 255.0
    return colors


def create_coordinate_frame(o3d_module, origin: np.ndarray, scale: float = 1.0):
    """
    Создает систему координат (оси X, Y, Z) для визуализации.
    
    Args:
        origin: точка начала координат
        scale: масштаб осей в метрах
        
    Returns:
        TriangleMesh с осями координат
    """
    # Создание координатных осей
    # X - красная, Y - зеленая, Z - синяя
    axes = o3d_module.geometry.TriangleMesh.create_coordinate_frame(
        size=scale,
        origin=origin
    )
    return axes


def visualize_tile(ply_path: str, stats_path: str = None, 
                   show_coords: bool = True,
                   coord_scale: float = None,
                   point_size: float = 1.0,
                   color_by: str = "rgb",
                   headless: bool = False):
    """
    Визуализирует тайл point cloud с системой координат.
    
    Args:
        ply_path: путь к PLY файлу тайла
        stats_path: путь к JSON файлу со статистикой
        show_coords: показывать ли систему координат
        coord_scale: масштаб осей координат (если None, вычисляется автоматически)
        point_size: размер точек для визуализации
    """
    print(f"Загрузка тайла: {ply_path}")
    
    # Загрузка данных
    points, labels, stats = load_tile(ply_path, stats_path)
    
    if points is None or len(points) == 0:
        print("Ошибка: тайл пуст или не может быть загружен")
        return
    
    # Извлечение координат и цветов
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    
    # Нормализация RGB в диапазон [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    class_rgb = class_labels_to_colors(labels)
    unique_labels = None
    if labels is not None:
        unique_labels = np.unique(labels.astype(np.int32))

    # Вывод статистики
    print("\n" + "="*60)
    print("СТАТИСТИКА ТАЙЛА")
    print("="*60)
    
    if stats:
        print(f"ID тайла: {stats.get('tile_id', 'N/A')}")
        print(f"Количество точек: {stats.get('num_points', len(xyz))}")
        
        coord_sys = stats.get('coordinate_system', {})
        print(f"\nСистема координат:")
        print(f"  Тип: {coord_sys.get('coordinate_type', 'N/A')}")
        center_global = coord_sys.get('center_global', coord_sys.get('origin_global', [0, 0, 0]))
        print(f"  Центр (глобальная система): ({center_global[0]:.2f}, {center_global[1]:.2f}, {center_global[2]:.2f})")
        center_local = coord_sys.get('center_local', coord_sys.get('origin_local', [0, 0, 0]))
        print(f"  Центр (локальная система): ({center_local[0]:.2f}, {center_local[1]:.2f}, {center_local[2]:.2f})")
        
        bounds = stats.get('bounds', {})
        print(f"\nГраницы (локальная система):")
        print(f"  X: [{bounds.get('x_min', 0):.2f}, {bounds.get('x_max', 0):.2f}]")
        print(f"  Y: [{bounds.get('y_min', 0):.2f}, {bounds.get('y_max', 0):.2f}]")
        print(f"  Z: [{bounds.get('z_min', 0):.2f}, {bounds.get('z_max', 0):.2f}]")
        
        center = stats.get('center', {})
        print(f"\nЦентр (локальная система):")
        print(f"  ({center.get('x', 0):.2f}, {center.get('y', 0):.2f}, {center.get('z', 0):.2f})")
        
        extent = stats.get('extent', {})
        print(f"\nРазмеры:")
        print(f"  X: {extent.get('x', 0):.2f} м")
        print(f"  Y: {extent.get('y', 0):.2f} м")
        print(f"  Z: {extent.get('z', 0):.2f} м")
        
        if 'label_distribution' in stats:
            print(f"\nРаспределение классов:")
            for label, count in stats['label_distribution'].items():
                cls_int = int(label)
                cls_name = SENSAT_CLASS_NAMES.get(cls_int, "Unknown")
                print(f"  Класс {cls_int:>2} ({cls_name}): {count} точек")
        if 'unique_classes' in stats:
            print(f"Уникальные классы (из stats): {stats['unique_classes']}")
        if 'labels_present_in_source' in stats:
            print(f"Метки были в исходном файле: {bool(stats['labels_present_in_source'])}")
    elif labels is not None:
        unique_labels, counts = np.unique(labels.astype(np.int32), return_counts=True)
        print("\nРаспределение классов:")
        for label, count in zip(unique_labels, counts):
            cls_name = SENSAT_CLASS_NAMES.get(int(label), "Unknown")
            print(f"  Класс {int(label):>2} ({cls_name}): {int(count)} точек")
    else:
        print(f"Количество точек: {len(xyz)}")
        print(f"Границы X: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
        print(f"Границы Y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
        print(f"Границы Z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")

    if unique_labels is not None:
        unique_as_list = [int(x) for x in unique_labels]
        unique_with_names = [f"{x}:{SENSAT_CLASS_NAMES.get(x, 'Unknown')}" for x in unique_as_list]
        print(f"Уникальные классы (из PLY): {unique_as_list}")
        print(f"Уникальные классы (с именами): {', '.join(unique_with_names)}")
        if stats and 'unique_classes' in stats:
            stats_unique = [int(x) for x in stats['unique_classes']]
            if sorted(stats_unique) != sorted(unique_as_list):
                print("ПРЕДУПРЕЖДЕНИЕ: unique_classes в stats и PLY не совпадают.")
    
    print("="*60)
    
    # Определение масштаба для осей координат
    if coord_scale is None:
        # Масштаб = 10% от максимального размера тайла
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        coord_scale = extent.max() * 0.1
        if coord_scale < 1.0:
            coord_scale = 1.0
    
    if headless:
        print("Режим headless: окно визуализации не открывается.")
        return

    # Отложенный импорт open3d: позволяет запускать --headless без GUI/OpenMP проблем.
    import open3d as o3d

    def show_scene(scene_colors: np.ndarray, mode_name: str) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(scene_colors)

        geometries = [pcd]
        if show_coords:
            # Центр тайла в локальной системе координат: X,Y в центре, Z на земле (0)
            # Система координат отображается в точке (0, 0, 0), где Z=0 - это уровень земли
            origin = np.array([0.0, 0.0, 0.0])
            coord_frame = create_coordinate_frame(o3d, origin, scale=coord_scale)
            geometries.append(coord_frame)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Tile Visualization: {Path(ply_path).name} [{mode_name}]")

        for geom in geometries:
            vis.add_geometry(geom)

        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        view_control = vis.get_view_control()
        view_control.set_front([0.5, -0.5, -0.7])
        view_control.set_lookat(xyz.mean(axis=0))
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(0.7)

        print(f"\nОтображение визуализации [{mode_name}]...")
        vis.run()
        vis.destroy_window()

    print("\nИнструкции:")
    print("  - Вращение: ЛКМ + движение мыши")
    print("  - Панорамирование: СКМ + движение мыши")
    print("  - Масштабирование: Колесо мыши")
    print("  - Закрытие: ESC или закрытие окна")

    if color_by == "class":
        print("\nСначала будет показан исходный RGB, затем классовая раскраска.")
        if stats and stats.get('labels_present_in_source') is False:
            print("ПРЕДУПРЕЖДЕНИЕ: в исходном файле не было class/cla, классовая раскраска будет однотонной.")
        show_scene(rgb, "rgb")
        if class_rgb is None:
            print("Предупреждение: в тайле нет class/cla, второй показ (class) пропущен.")
        else:
            show_scene(class_rgb, "class")
    else:
        show_scene(rgb, "rgb")


def main():
    parser = argparse.ArgumentParser(
        description='Визуализация тайла point cloud с системой координат'
    )
    parser.add_argument('--tile', type=str, required=True,
                       help='Путь к PLY файлу тайла')
    parser.add_argument('--stats', type=str, default=None,
                       help='Путь к JSON файлу со статистикой (если не указан, будет найден автоматически)')
    parser.add_argument('--no-coords', action='store_true',
                       help='Не показывать систему координат')
    parser.add_argument('--coord-scale', type=float, default=None,
                       help='Масштаб осей координат в метрах (по умолчанию: автоматический)')
    parser.add_argument('--point-size', type=float, default=1.0,
                       help='Размер точек для визуализации (по умолчанию: 1.0)')
    parser.add_argument('--color-by', type=str, default='rgb', choices=['rgb', 'class'],
                       help='Режим раскраски точек: rgb или class (по умолчанию: rgb)')
    parser.add_argument('--headless', action='store_true',
                       help='Проверка загрузки/статистики без открытия GUI-окна')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tile):
        print(f"Ошибка: файл {args.tile} не найден")
        return
    
    visualize_tile(
        args.tile,
        stats_path=args.stats,
        show_coords=not args.no_coords,
        coord_scale=args.coord_scale,
        point_size=args.point_size,
        color_by=args.color_by,
        headless=args.headless
    )


if __name__ == '__main__':
    main()
