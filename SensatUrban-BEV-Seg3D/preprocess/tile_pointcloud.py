#!/usr/bin/env python3
"""
Скрипт для нарезки point cloud на тайлы с сохранением статистики и системы координат.

Каждый тайл имеет свою локальную систему координат (смещенную относительно исходной).
Статистика сохраняется в JSON файл для каждого тайла.

Примеры использования:
    # Обработка одного файла
    python tile_pointcloud.py --input data.ply --output tiles/ --tile-size 50 --tile-step 50
    
    # Обработка директории с файлами
    python tile_pointcloud.py --input data_dir/ --output tiles/ --tile-size 50 --tile-step 50 --num-workers 4
    
    # Режим отладки (только первый тайл)
    python tile_pointcloud.py --input data.ply --output tiles/ --debug
    
    # С настройками минимального количества точек
    python tile_pointcloud.py --input data.ply --output tiles/ --min-points 2000
"""

import os
import json
import argparse
import math
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Импорт вспомогательных функций
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_ply import read_ply, write_ply


def calculate_tile_statistics(xyz: np.ndarray, rgb: np.ndarray, 
                             labels: Optional[np.ndarray] = None,
                             tile_origin: np.ndarray = None) -> Dict:
    """
    Вычисляет статистику для тайла.
    
    Args:
        xyz: координаты точек (N, 3)
        rgb: цвета точек (N, 3)
        labels: метки классов (N,) или None
        tile_origin: исходные координаты начала тайла в глобальной системе
        
    Returns:
        Словарь со статистикой
    """
    stats = {
        'num_points': int(len(xyz)),
        'coordinate_system': {
            'center_global': tile_origin.tolist() if tile_origin is not None else [0.0, 0.0, 0.0],
            'center_local': [0.0, 0.0, 0.0],  # Центр тайла в локальной системе: X,Y в центре, Z на земле (0)
            'coordinate_type': 'local_centered_ground'  # X,Y центрированы, Z на уровне земли (min Z = 0)
        },
        'bounds': {
            'x_min': float(xyz[:, 0].min()),
            'x_max': float(xyz[:, 0].max()),
            'y_min': float(xyz[:, 1].min()),
            'y_max': float(xyz[:, 1].max()),
            'z_min': float(xyz[:, 2].min()),
            'z_max': float(xyz[:, 2].max())
        },
        'center': {
            'x': float(xyz[:, 0].mean()),
            'y': float(xyz[:, 1].mean()),
            'z': float(xyz[:, 2].mean())
        },
        'extent': {
            'x': float(xyz[:, 0].max() - xyz[:, 0].min()),
            'y': float(xyz[:, 1].max() - xyz[:, 1].min()),
            'z': float(xyz[:, 2].max() - xyz[:, 2].min())
        },
        'rgb_stats': {
            'r_min': int(rgb[:, 0].min()) if rgb is not None else None,
            'r_max': int(rgb[:, 0].max()) if rgb is not None else None,
            'r_mean': float(rgb[:, 0].mean()) if rgb is not None else None,
            'g_min': int(rgb[:, 1].min()) if rgb is not None else None,
            'g_max': int(rgb[:, 1].max()) if rgb is not None else None,
            'g_mean': float(rgb[:, 1].mean()) if rgb is not None else None,
            'b_min': int(rgb[:, 2].min()) if rgb is not None else None,
            'b_max': int(rgb[:, 2].max()) if rgb is not None else None,
            'b_mean': float(rgb[:, 2].mean()) if rgb is not None else None
        }
    }
    
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats['label_distribution'] = {
            int(label): int(count) for label, count in zip(unique_labels, counts)
        }
        stats['num_classes'] = int(len(unique_labels))
        stats['most_common_class'] = int(unique_labels[np.argmax(counts)])
    
    return stats


def export_pointcept_tile(
    tile_id: str,
    xyz_local: np.ndarray,
    rgb: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    pointcept_split_dir: str,
) -> str:
    """
    Экспортирует тайл в формат Pointcept DefaultDataset:
      <split>/<tile_id>/coord.npy,color.npy,normal.npy,segment.npy
    """
    sample_dir = os.path.join(pointcept_split_dir, tile_id)
    os.makedirs(sample_dir, exist_ok=True)

    coord = xyz_local.astype(np.float32, copy=False)
    if rgb is None:
        color = np.zeros_like(coord, dtype=np.float32)
    else:
        color = rgb.astype(np.float32, copy=False)

    # Заглушка нормалей: при необходимости можно заменить оценкой нормалей.
    normal = np.zeros_like(coord, dtype=np.float32)

    if labels is None:
        segment = np.full((coord.shape[0],), -1, dtype=np.int32)
    else:
        segment = labels.reshape(-1).astype(np.int32, copy=False)

    np.save(os.path.join(sample_dir, "coord.npy"), coord)
    np.save(os.path.join(sample_dir, "color.npy"), color)
    np.save(os.path.join(sample_dir, "normal.npy"), normal)
    np.save(os.path.join(sample_dir, "segment.npy"), segment)

    return sample_dir


def process_single_tile(args: Tuple) -> Optional[Dict]:
    """
    Обрабатывает один тайл из point cloud.
    
    Args:
        args: кортеж (x_idx, y_idx, tile_points, tile_size, output_dir, base_name,
            min_points_per_tile, tile_origin_global, labels_present_in_source,
            export_pointcept, pointcept_split_dir)
    
    Returns:
        Словарь с информацией о тайле или None если тайл не прошел фильтрацию
    """
    (x_idx, y_idx, tile_points, tile_size, output_dir,
     base_name, min_points_per_tile, tile_origin_global, labels_present_in_source,
     export_pointcept, pointcept_split_dir) = args
    
    if tile_points is None or len(tile_points) == 0:
        return None
    
    # Извлечение данных
    xyz = tile_points[:, :3]
    rgb = tile_points[:, 3:6] if tile_points.shape[1] >= 6 else None
    labels = tile_points[:, 6] if tile_points.shape[1] >= 7 else None
    if labels is not None:
        # Явно сохраняем классы как целые значения.
        labels = labels.astype(np.int32, copy=False)
    
    # Проверка минимального количества точек
    if len(xyz) < min_points_per_tile:
        return None
    
    # Вычисление центра тайла в глобальных координатах на основе сетки тайлов
    # X и Y - центр тайла по индексам сетки, Z - минимум (земля)
    if tile_origin_global is not None:
        tile_center_global = np.array(tile_origin_global)
    else:
        tile_center_global = np.array([
            x_idx + tile_size * 0.5,
            y_idx + tile_size * 0.5,
            xyz[:, 2].min()
        ])
    
    # Смещение координат в локальную систему
    # X и Y смещаются к центру, Z смещается так, чтобы минимум был 0 (земля)
    xyz_local = xyz.copy()
    xyz_local[:, 0] -= tile_center_global[0]  # Смещаем X к центру
    xyz_local[:, 1] -= tile_center_global[1]  # Смещаем Y к центру
    xyz_local[:, 2] -= tile_center_global[2]  # Смещаем Z так, чтобы минимум был 0 (земля)
    
    # Глобальный центр тайла (для восстановления координат)
    # X, Y - центр, Z - уровень земли
    tile_center_global_actual = tile_center_global.copy()
    
    # Вычисление статистики
    stats = calculate_tile_statistics(
        xyz_local, rgb, labels, 
        tile_origin=tile_center_global_actual
    )
    
    # Формирование имени файла
    tile_id = f"{base_name}_tile_{x_idx}_{y_idx}"
    tile_ply_path = os.path.join(output_dir, f"{tile_id}.ply")
    tile_stats_path = os.path.join(output_dir, f"{tile_id}_stats.json")
    
    # Подготовка данных для сохранения
    if rgb is not None:
        if labels is not None:
            field_list = [xyz_local, rgb, labels.reshape(-1, 1)]
            field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'class']
        else:
            field_list = [xyz_local, rgb]
            field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    else:
        if labels is not None:
            field_list = [xyz_local, labels.reshape(-1, 1)]
            field_names = ['x', 'y', 'z', 'class']
        else:
            field_list = [xyz_local]
            field_names = ['x', 'y', 'z']
    
    # Сохранение PLY файла
    try:
        write_ply(tile_ply_path, field_list, field_names)
    except Exception as e:
        print(f"Ошибка при сохранении {tile_ply_path}: {e}")
        return None
    
    # Сохранение статистики
    stats['tile_id'] = tile_id
    stats['tile_index'] = {'x': int(x_idx), 'y': int(y_idx)}
    stats['tile_size'] = float(tile_size)
    stats['labels_present_in_source'] = bool(labels_present_in_source)
    stats['labels_field'] = 'class' if labels is not None else None
    if labels is not None:
        stats['unique_classes'] = [int(x) for x in np.unique(labels)]

    with open(tile_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    pointcept_sample_dir = None
    if export_pointcept and pointcept_split_dir:
        pointcept_sample_dir = export_pointcept_tile(
            tile_id=tile_id,
            xyz_local=xyz_local,
            rgb=rgb,
            labels=labels,
            pointcept_split_dir=pointcept_split_dir,
        )
    
    return {
        'tile_id': tile_id,
        'tile_path': tile_ply_path,
        'stats_path': tile_stats_path,
        'pointcept_sample_dir': pointcept_sample_dir,
        'num_points': len(xyz),
        'x_idx': int(x_idx),
        'y_idx': int(y_idx)
    }


def grid_generator(ply_data: np.ndarray, grid_size: float, grid_step: float, 
                  margin: bool = False):
    """
    Генерирует тайлы из point cloud с использованием скользящего окна.
    
    Args:
        ply_data: массив точек (N, 7) [x, y, z, r, g, b, class]
        grid_size: размер тайла в метрах
        grid_step: шаг тайла в метрах
        margin: использовать ли margin для последних тайлов
        
    Yields:
        (x_idx, y_idx, tile_points) для каждого тайла
    """
    if ply_data is None or len(ply_data) == 0:
        return
    
    margin = 1 if margin else 0
    
    # Сортировка по X
    idx_sort_x = np.argsort(ply_data[:, 0])
    ply_sort_x = ply_data[idx_sort_x, :]
    x_max = math.ceil(ply_sort_x[-1, 0])
    x_min = math.floor(ply_sort_x[0, 0])
    
    for idx_start_x in range(x_min, x_max + margin, int(grid_step)):
        # Фильтрация по X
        x_mask = (ply_sort_x[:, 0] >= idx_start_x) & (ply_sort_x[:, 0] < idx_start_x + grid_size)
        grid_sort_x = ply_sort_x[x_mask, :].copy()
        
        if grid_sort_x is None or len(grid_sort_x) == 0:
            continue
        
        # Сортировка по Y
        idx_sort_y = np.argsort(grid_sort_x[:, 1])
        grid_sort_x = grid_sort_x[idx_sort_y, :]
        y_max = math.ceil(grid_sort_x[-1, 1])
        y_min = math.floor(grid_sort_x[0, 1])
        
        for idx_start_y in range(y_min, y_max + margin, int(grid_step)):
            # Фильтрация по Y
            y_mask = (grid_sort_x[:, 1] >= idx_start_y) & (grid_sort_x[:, 1] < idx_start_y + grid_size)
            grid_sort_xy = grid_sort_x[y_mask, :].copy()
            
            if grid_sort_xy is None or len(grid_sort_xy) == 0:
                continue
            
            # Не смещаем координаты здесь - это будет сделано в process_single_tile
            # для правильной обработки локальной системы координат
            
            yield (idx_start_x, idx_start_y, grid_sort_xy)
        
        # Обновление для следующей итерации по X
        ply_sort_x = ply_sort_x[~x_mask, :]


def process_ply_file(ply_path: str, output_dir: str, 
                    tile_size: float = 50.0,
                    tile_step: float = 50.0,
                    min_points_per_tile: int = 1000,
                    num_workers: int = 1,
                    parallel_backend: str = "process",
                    chunksize: int = 0,
                    export_pointcept: bool = False,
                    pointcept_split_dir: Optional[str] = None,
                    debug: bool = False) -> List[Dict]:
    """
    Обрабатывает один PLY файл, нарезая его на тайлы.
    
    Args:
        ply_path: путь к PLY файлу
        output_dir: директория для сохранения тайлов
        tile_size: размер тайла в метрах
        tile_step: шаг тайла в метрах
        min_points_per_tile: минимальное количество точек в тайле
        num_workers: количество процессов для параллельной обработки
        parallel_backend: backend распараллеливания: process или thread
        chunksize: размер чанка для process backend (0 = авто)
        export_pointcept: экспортировать ли каждый тайл в формат Pointcept
        pointcept_split_dir: директория split в формате Pointcept
        debug: режим отладки (обрабатывает только первый тайл)
        
    Returns:
        Список словарей с информацией о созданных тайлах
    """
    print(f"Загрузка файла: {ply_path}")
    
    # Загрузка данных
    ply_data = read_ply(ply_path)
    
    # Преобразование в массив
    x = ply_data["x"]
    y = ply_data["y"]
    z = ply_data["z"]
    
    # RGB
    if "red" in ply_data.dtype.names:
        r = ply_data["red"]
        g = ply_data["green"]
        b = ply_data["blue"]
    elif "alpha" in ply_data.dtype.names:
        # Если нет RGB, используем alpha как grayscale
        a = ply_data["alpha"]
        r = a.copy()
        g = a.copy()
        b = a.copy()
    else:
        r = np.zeros_like(x, dtype=np.uint8)
        g = np.zeros_like(x, dtype=np.uint8)
        b = np.zeros_like(x, dtype=np.uint8)
    
    # Labels
    labels_present_in_source = False
    if "class" in ply_data.dtype.names:
        labels = ply_data["class"]
        labels_present_in_source = True
    elif "cla" in ply_data.dtype.names:
        labels = ply_data["cla"]
        labels_present_in_source = True
    else:
        print(f"ВНИМАНИЕ: в исходном файле нет class/cla, будет использован класс 0 для всех точек: {ply_path}")
        labels = np.zeros_like(x, dtype=np.uint8)
    
    # Формирование массива [x, y, z, r, g, b, class]
    points = np.vstack((x, y, z, r, g, b, labels)).T
    
    # Базовое имя файла
    base_name = Path(ply_path).stem
    
    # Создание директории для тайлов
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерация тайлов
    print("Генерация тайлов...")
    tiles_data = []
    grid_gen = grid_generator(points, tile_size, tile_step, margin=False)
    
    for x_idx, y_idx, tile_points in grid_gen:
        tiles_data.append((x_idx, y_idx, tile_points, tile_size, output_dir, 
                          base_name, min_points_per_tile, None, labels_present_in_source,
                          export_pointcept, pointcept_split_dir))
        
        if debug:
            # В режиме отладки обрабатываем только первый тайл
            break
    
    print(f"Найдено {len(tiles_data)} тайлов для обработки")
    
    # Обработка тайлов
    if num_workers > 1 and len(tiles_data) > 1:
        if parallel_backend == "thread":
            print(f"Параллельная обработка с {num_workers} потоками...")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(process_single_tile, tiles_data),
                    total=len(tiles_data),
                    desc="Обработка тайлов"
                ))
        else:
            actual_chunksize = chunksize if chunksize and chunksize > 0 else max(1, len(tiles_data) // (num_workers * 8))
            print(f"Параллельная обработка с {num_workers} процессами (chunksize={actual_chunksize})...")
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_tile, tiles_data, chunksize=actual_chunksize),
                    total=len(tiles_data),
                    desc="Обработка тайлов"
                ))
    else:
        print("Последовательная обработка...")
        results = []
        for args in tqdm(tiles_data, desc="Обработка тайлов"):
            results.append(process_single_tile(args))
    
    # Фильтрация None результатов
    results = [r for r in results if r is not None]
    
    print(f"Создано {len(results)} тайлов")
    
    # Сохранение общего индекса тайлов
    index_path = os.path.join(output_dir, f"{base_name}_tiles_index.json")
    with open(index_path, 'w') as f:
        json.dump({
            'source_file': ply_path,
            'tile_size': tile_size,
            'tile_step': tile_step,
            'num_tiles': len(results),
            'tiles': [
                {
                    'tile_id': r['tile_id'],
                    'x_idx': r['x_idx'],
                    'y_idx': r['y_idx'],
                    'num_points': r['num_points']
                }
                for r in results
            ]
        }, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Нарезка point cloud на тайлы с сохранением статистики'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Путь к входному PLY файлу или директории с PLY файлами')
    parser.add_argument('--output', type=str, required=True,
                       help='Директория для сохранения тайлов')
    parser.add_argument('--split-name', type=str, default=None,
                       help='Имя split-папки под output (напр. train). Для директории по умолчанию берется имя входной папки')
    parser.add_argument('--group-by-source', action='store_true',
                       help='Сохранять тайлы в подпапках по исходным файлам (по умолчанию все тайлы в одной split-папке)')
    parser.add_argument('--clean-output-split', action='store_true',
                       help='Очистить output/<split> перед запуском')
    parser.add_argument('--tile-size', type=float, default=50.0,
                       help='Размер тайла в метрах (по умолчанию: 50.0)')
    parser.add_argument('--tile-step', type=float, default=50.0,
                       help='Шаг тайла в метрах (по умолчанию: 50.0)')
    parser.add_argument('--min-points', type=int, default=1000,
                       help='Минимальное количество точек в тайле (по умолчанию: 1000)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Количество процессов для параллельной обработки (по умолчанию: 1)')
    parser.add_argument('--parallel-backend', type=str, default='process', choices=['process', 'thread'],
                       help='Backend распараллеливания: process или thread (по умолчанию: process)')
    parser.add_argument('--chunksize', type=int, default=0,
                       help='Размер чанка для process backend (0 = авто)')
    parser.add_argument('--export-pointcept', action='store_true',
                       help='Дополнительно экспортировать тайлы в формат Pointcept (coord/color/normal/segment .npy)')
    parser.add_argument('--debug', action='store_true',
                       help='Режим отладки: обрабатывает только первый тайл из первого файла')
    
    args = parser.parse_args()
    
    # Определение входных файлов
    input_path = Path(args.input)
    if input_path.is_file():
        ply_files = [str(input_path)]
        split_name = args.split_name if args.split_name else "single"
    elif input_path.is_dir():
        ply_files = sorted(list(input_path.glob('*.ply')))
        split_name = args.split_name if args.split_name else input_path.name
    else:
        print(f"Ошибка: {args.input} не является файлом или директорией")
        return
    
    if len(ply_files) == 0:
        print(f"Не найдено PLY файлов в {args.input}")
        return
    
    print(f"Найдено {len(ply_files)} PLY файлов")
    
    if args.debug:
        print("РЕЖИМ ОТЛАДКИ: будет обработан только первый тайл из первого файла")
        ply_files = ply_files[:1]

    split_output_dir = os.path.join(args.output, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    if args.clean_output_split and os.path.exists(split_output_dir):
        print(f"Очистка split-директории: {split_output_dir}")
        shutil.rmtree(split_output_dir)
        os.makedirs(split_output_dir, exist_ok=True)
    print(f"Выходной split: {split_output_dir}")
    
    # Обработка файлов
    all_results = []
    for ply_file in ply_files:
        print(f"\n{'='*60}")
        print(f"Обработка: {ply_file}")
        print(f"{'='*60}")
        
        # Сохраняем либо в split-папку, либо в подпапку по исходному файлу
        file_output_dir = split_output_dir
        if args.group_by_source:
            file_output_dir = os.path.join(split_output_dir, Path(ply_file).stem)

        results = process_ply_file(
            str(ply_file),
            file_output_dir,
            tile_size=args.tile_size,
            tile_step=args.tile_step,
            min_points_per_tile=args.min_points,
            num_workers=args.num_workers,
            parallel_backend=args.parallel_backend,
            chunksize=args.chunksize,
            export_pointcept=args.export_pointcept,
            pointcept_split_dir=split_output_dir if args.export_pointcept else None,
            debug=args.debug
        )
        
        all_results.extend(results)
        
        if args.debug:
            break
    
    print(f"\n{'='*60}")
    print(f"Обработка завершена!")
    print(f"Всего создано тайлов: {len(all_results)}")
    print(f"Результаты сохранены в: {split_output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
