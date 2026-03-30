#!/usr/bin/env python3
"""
Рендерит вращающуюся 3D модель в GIF для презентации/защиты.

Камера вращается вокруг центра модели. Совместимо с Open3D >= 0.14.

Usage:
    python render_gif.py \
        --ply pred.ply \
        --out rotation.gif \
        --frames 90 \
        --fps 20 \
        --elevation 35
"""

import argparse
import math
import os
import tempfile

import numpy as np
import open3d as o3d
from PIL import Image


def render_rotation_gif(
    ply_path: str,
    out_path: str,
    n_frames: int = 90,
    width: int = 1280,
    height: int = 960,
    fps: int = 20,
    elevation_deg: float = 30.0,
    zoom: float = 0.7,
    bg_color: tuple = (1.0, 1.0, 1.0),
    point_size: float = 2.0,
):
    print(f"Loading: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    n_pts = len(pcd.points)
    print(f"  Points: {n_pts:,}")

    # Центрируем модель в начало координат — камера будет вращаться вокруг (0,0,0)
    center = pcd.get_center()
    pcd.translate(-center)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    camera_dist = extent / zoom
    elev_rad = math.radians(elevation_deg)

    # Создаём Visualizer, ставим начальный вид, рендерим кадры
    # Вместо изменения extrinsic — поворачиваем саму геометрию,
    # а камеру фиксируем. Это надёжно работает во всех версиях Open3D.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.point_size = point_size

    # Фиксируем камеру: смотрит с elevation вдоль оси Y
    ctr = vis.get_view_control()
    ctr.set_front([0.0, -math.cos(elev_rad), math.sin(elev_rad)])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    angle_step = 2.0 * math.pi / n_frames

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_frames):
            vis.poll_events()
            vis.update_renderer()

            img_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            vis.capture_screen_image(img_path, do_render=True)
            frames.append(Image.open(img_path).copy())

            # Поворачиваем облако точек вокруг оси Z на один шаг
            R = pcd.get_rotation_matrix_from_xyz((0, 0, angle_step))
            pcd.rotate(R, center=(0, 0, 0))
            vis.update_geometry(pcd)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Frame {i + 1}/{n_frames}")

    vis.destroy_window()

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB, {n_frames} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Рендер вращающейся 3D модели в GIF"
    )
    parser.add_argument("--ply", type=str, required=True,
                        help="Путь к PLY файлу")
    parser.add_argument("--out", type=str, required=True,
                        help="Путь к выходному GIF")
    parser.add_argument("--frames", type=int, default=90,
                        help="Количество кадров (default: 90)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Кадров в секунду (default: 20)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Ширина кадра (default: 1280)")
    parser.add_argument("--height", type=int, default=960,
                        help="Высота кадра (default: 960)")
    parser.add_argument("--elevation", type=float, default=30.0,
                        help="Угол подъёма камеры в градусах (default: 30)")
    parser.add_argument("--zoom", type=float, default=0.7,
                        help="Зум (меньше = дальше, default: 0.7)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Размер точки (default: 2.0)")
    parser.add_argument("--bg", type=str, default="white",
                        choices=["white", "black"],
                        help="Цвет фона (default: white)")

    args = parser.parse_args()
    bg = (1.0, 1.0, 1.0) if args.bg == "white" else (0.0, 0.0, 0.0)

    render_rotation_gif(
        ply_path=args.ply,
        out_path=args.out,
        n_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        elevation_deg=args.elevation,
        zoom=args.zoom,
        bg_color=bg,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
