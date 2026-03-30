#!/usr/bin/env python3
"""
Скрипт для извлечения всех уникальных цветов из изображения.

Полезно для анализа масок сегментации и настройки конфигурации.
"""

import argparse
import cv2
import numpy as np
from collections import Counter


def extract_unique_colors(image_path: str, min_pixels: int = 1, 
                          color_space: str = 'BGR', sort_by_count: bool = False):
    """
    Извлекает все уникальные цвета из изображения.
    
    Args:
        image_path: путь к изображению
        min_pixels: минимальное количество пикселей для вывода цвета
        color_space: 'BGR' или 'RGB' - в каком формате выводить цвета
        sort_by_count: сортировать ли по количеству пикселей
    
    Returns:
        Список кортежей (цвет, количество пикселей)
    """
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Преобразуем в RGB если нужно
    if color_space == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Получаем все пиксели
    pixels = img.reshape(-1, 3)
    
    # Подсчитываем уникальные цвета
    # Преобразуем в кортежи для использования в Counter
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    color_counts = Counter(pixel_tuples)
    
    # Фильтруем по минимальному количеству пикселей
    filtered_colors = [(color, count) for color, count in color_counts.items() 
                      if count >= min_pixels]
    
    # Сортируем
    if sort_by_count:
        filtered_colors.sort(key=lambda x: x[1], reverse=True)
    else:
        filtered_colors.sort()
    
    return filtered_colors


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение уникальных цветов из изображения'
    )
    
    parser.add_argument('image', type=str,
                       help='Путь к изображению')
    parser.add_argument('--min_pixels', type=int, default=1,
                       help='Минимальное количество пикселей для вывода цвета (по умолчанию: 1)')
    parser.add_argument('--color_space', type=str, default='BGR', choices=['BGR', 'RGB'],
                       help='Цветовое пространство для вывода (по умолчанию: BGR)')
    parser.add_argument('--sort_by_count', action='store_true',
                       help='Сортировать по количеству пикселей (по убыванию)')
    parser.add_argument('--format', type=str, default='list', 
                       choices=['list', 'yaml', 'json'],
                       help='Формат вывода (по умолчанию: list)')
    parser.add_argument('--show_stats', action='store_true',
                       help='Показывать статистику (количество пикселей)')
    
    args = parser.parse_args()
    
    # Извлекаем цвета
    colors = extract_unique_colors(
        args.image, 
        args.min_pixels, 
        args.color_space,
        args.sort_by_count
    )
    
    total_pixels = sum(count for _, count in colors)
    
    # Выводим результаты
    if args.format == 'list':
        print(f"Найдено уникальных цветов: {len(colors)}")
        print(f"Всего пикселей: {total_pixels}")
        print("\nЦвета:")
        print("-" * 60)
        for color, count in colors:
            color_str = f"[{color[0]}, {color[1]}, {color[2]}]"
            if args.show_stats:
                percentage = (count / total_pixels) * 100
                print(f"{color_str:20} - {count:8} пикселей ({percentage:5.2f}%)")
            else:
                print(color_str)
    
    elif args.format == 'yaml':
        print("# Уникальные цвета из изображения")
        print(f"# Всего цветов: {len(colors)}, всего пикселей: {total_pixels}")
        print("colors:")
        for color, count in colors:
            color_list = f"[{color[0]}, {color[1]}, {color[2]}]"
            if args.show_stats:
                percentage = (count / total_pixels) * 100
                print(f"  - color: {color_list}  # {count} пикселей ({percentage:.2f}%)")
            else:
                print(f"  - {color_list}")
    
    elif args.format == 'json':
        import json
        colors_data = []
        for color, count in colors:
            color_dict = {
                'color': list(color),
                'count': count
            }
            if args.show_stats:
                color_dict['percentage'] = (count / total_pixels) * 100
            colors_data.append(color_dict)
        
        output = {
            'total_colors': len(colors),
            'total_pixels': total_pixels,
            'colors': colors_data
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
