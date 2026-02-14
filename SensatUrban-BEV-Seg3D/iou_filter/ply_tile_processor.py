#!/usr/bin/env python3
# ply_tile_processor.py
# Class for processing PLY files, creating BEV projections, and tiling

import os
import math
import numpy as np
import cv2
from typing import Tuple, List, Iterator, Optional
from tqdm import tqdm

import sys
preprocess_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess')
sys.path.insert(0, preprocess_path)
from helper_ply import read_ply


def project_3d_bev_img3(grid_data, grid_scale, grid_size, label_color_map, imgid=None):
    """
    Project 3D points to 2D BEV pixels (BEV Projection)

    IMPORTANT FIX:
    We store into arrays using image convention: [row=y, col=x].
    So first axis is y (height), second axis is x (width).
    """
    grid_size_scale = int(grid_size / grid_scale)
    num = grid_data.shape[1]

    # altitude + rgb (3 channels). Use float for intermediate
    bev = np.zeros((grid_size_scale, grid_size_scale, 4), dtype=np.float32) - 1000
    cla = np.zeros((grid_size_scale, grid_size_scale), dtype=np.float32) - 1000

    xs = (grid_data[0] / grid_scale).astype(np.int32)  # x -> col
    ys = (grid_data[1] / grid_scale).astype(np.int32)  # y -> row

    # clip just in case numerical edge puts a point on border
    xs = np.clip(xs, 0, grid_size_scale - 1)
    ys = np.clip(ys, 0, grid_size_scale - 1)

    # write as [y, x]
    for i in range(num):
        if bev[ys[i], xs[i], 0] < grid_data[2, i]:
            bev[ys[i], xs[i], 0] = grid_data[2, i]      # altitude
            bev[ys[i], xs[i], 1] = grid_data[3, i]      # R
            bev[ys[i], xs[i], 2] = grid_data[4, i]      # G
            bev[ys[i], xs[i], 3] = grid_data[5, i]      # B
            cla[ys[i], xs[i]] = grid_data[6, i]         # class

    alt = bev[:, :, 0]
    rgb = bev[:, :, 1:4]  # (H, W, 3) in RGB order
    return alt, rgb, cla


def complete2d(src_map, loops, keep_margin=False):
    """Completion for 2d image - simplified version"""
    for _ in range(loops):
        kernel = np.ones((3, 3), np.uint8)
        comp_map = cv2.morphologyEx(src_map, cv2.MORPH_CLOSE, kernel)
        invalid_idx = (src_map == -1000)
        src_map[invalid_idx] = comp_map[invalid_idx]
    return src_map


class PLYTileProcessor:
    """
    Process PLY files, create BEV projections, and handle tiling.
    Similar to SensatUrbanEDA from point_EDA_31.py
    """

    def __init__(self, grid_scale: float = 0.05, grid_size: float = 25, grid_step: int = 25):
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.grid_step = grid_step

        self.label_color_map = [
            [255, 248, 220],  # 0: Ground
            [220, 220, 220],  # 1: Vegetation
            [139, 71, 38],    # 2: Building
            [238, 197, 145],  # 3: Wall
            [70, 130, 180],   # 4: Bridge
            [179, 238, 58],   # 5: Parking
            [110, 139, 61],   # 6: Rail
            [105, 105, 105],  # 7: Traffic
            [0, 0, 128],      # 8: Street
            [205, 92, 92],    # 9: Car
            [244, 164, 96],   # 10: Footpath
            [147, 112, 219],  # 11: Bike
            [255, 228, 225],  # 12: Water
        ]
        self.label_color_map_new = [[0, 0, 0]] + self.label_color_map

    def load_ply(self, ply_path: str, reformat: bool = True) -> np.ndarray:
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        _ply_data = read_ply(ply_path)

        if reformat:
            x = _ply_data["x"]
            y = _ply_data["y"]
            z = _ply_data["z"]
            r = _ply_data["red"]
            g = _ply_data["green"]
            b = _ply_data["blue"]
            c = _ply_data["class"]
            _ply_data = np.vstack((x, y, z, r, g, b, c)).T  # (N, 7)

        return _ply_data

    def project_bev(self, grid_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure shape is (7, N)
        if grid_data.shape[0] != 7:
            grid_data = grid_data.T

        alt, rgb, cla = project_3d_bev_img3(
            grid_data, self.grid_scale, self.grid_size,
            self.label_color_map, imgid=None
        )

        # Completion to fill holes
        n_loop = 3
        alt = complete2d(alt, n_loop)
        cla = complete2d(cla, n_loop)
        for c in range(3):
            rgb[:, :, c] = complete2d(rgb[:, :, c], n_loop)

        # Convert rgb to uint8 safely:
        # invalid stays -1000; after completion should be filled, but clip anyway
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Relabel: -1000 -> -1, then +1 to shift to [0, 13]
        cla[cla == -1000] = -1
        cla = cla + 1  # 0 = unlabeled

        return alt, rgb, cla.astype(np.int64)

    def grid_generator(self, ply_data: np.ndarray, margin: bool = False) -> Iterator[Tuple[int, int, np.ndarray]]:
        if ply_data is None or ply_data.size == 0:
            raise ValueError("Empty PLY data")

        margin_val = 1 if margin else 0

        idx_sort_x = np.argsort(ply_data[:, 0])
        ply_sort_x = ply_data[idx_sort_x, :]
        x_max = math.ceil(ply_sort_x[-1, 0])
        x_min = math.floor(ply_sort_x[0, 0])

        for idx_start_x in tqdm(range(x_min, x_max + margin_val, self.grid_step), desc="Generating tiles"):
            idx_grid_x = np.where((ply_sort_x[:, 0] >= idx_start_x) &
                                  (ply_sort_x[:, 0] < idx_start_x + self.grid_size))[0]
            grid_sort_x = ply_sort_x[idx_grid_x, :]
            if grid_sort_x is None or grid_sort_x.size == 0:
                continue

            idx_sort_y = np.argsort(grid_sort_x[:, 1])
            grid_sort_x = grid_sort_x[idx_sort_y, :]
            y_max = math.ceil(grid_sort_x[-1, 1])
            y_min = math.floor(grid_sort_x[0, 1])

            for idx_start_y in range(y_min, y_max + margin_val, self.grid_step):
                idx_grid_xy = np.where((grid_sort_x[:, 1] >= idx_start_y) &
                                       (grid_sort_x[:, 1] < idx_start_y + self.grid_size))[0]
                grid_sort_xy = grid_sort_x[idx_grid_xy, :].copy()
                if grid_sort_xy is None or grid_sort_xy.size == 0:
                    continue

                # Normalize to local coords
                grid_sort_xy[:, :2] -= np.array([idx_start_x, idx_start_y])
                yield (idx_start_x, idx_start_y, grid_sort_xy)

    def process_ply_to_bev(self, ply_path: str, temp_dir: str, max_tiles: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        os.makedirs(temp_dir, exist_ok=True)

        print(f"[load] Loading PLY from: {ply_path}")
        ply_data = self.load_ply(ply_path, reformat=True)

        x_min = ply_data[:, 0].min()
        y_min = ply_data[:, 1].min()
        ply_data[:, 0] -= x_min
        ply_data[:, 1] -= y_min

        x_max = ply_data[:, 0].max()
        y_max = ply_data[:, 1].max()
        full_size = math.ceil(max(x_max, y_max) + self.grid_size)

        full_size_pixels = int(full_size / self.grid_scale)
        tile_size_pixels = int(self.grid_size / self.grid_scale)

        print(f"[bev] Full size: {full_size}m -> {full_size_pixels}px")
        print(f"[bev] Tile size: {self.grid_size}m -> {tile_size_pixels}px")

        full_rgb = np.zeros((full_size_pixels, full_size_pixels, 3), dtype=np.uint8)
        full_gt = np.zeros((full_size_pixels, full_size_pixels), dtype=np.int64)

        tiles_info = []
        grid_gen = self.grid_generator(ply_data, margin=False)

        tile_count = 0
        for x_idx, y_idx, grid_points in grid_gen:
            if max_tiles is not None and tile_count >= max_tiles:
                print(f"[debug] Limited to {max_tiles} tiles")
                break
            tile_count += 1

            alt, rgb, cla = self.project_bev(grid_points)

            # x -> col, y -> row
            x_px = int(x_idx / self.grid_scale)
            y_px = int(y_idx / self.grid_scale)

            tile_h = min(tile_size_pixels, full_size_pixels - y_px)
            tile_w = min(tile_size_pixels, full_size_pixels - x_px)

            rgb_tile = rgb[:tile_h, :tile_w]
            cla_tile = cla[:tile_h, :tile_w]

            tile_id = f"{x_idx}_{y_idx}"
            rgb_path = os.path.join(temp_dir, f"rgb_{tile_id}.png")
            gt_path = os.path.join(temp_dir, f"gt_{tile_id}.png")

            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(gt_path, cla_tile.astype(np.uint8))

            full_rgb[y_px:y_px+tile_h, x_px:x_px+tile_w] = rgb_tile
            full_gt[y_px:y_px+tile_h, x_px:x_px+tile_w] = cla_tile

            tiles_info.append({
                'x_idx': x_idx,
                'y_idx': y_idx,
                'x_px': x_px,
                'y_px': y_px,
                'tile_h': tile_h,
                'tile_w': tile_w,
                'rgb_path': rgb_path,
                'gt_path': gt_path,
            })

        metadata = {
            'full_size': full_size,
            'full_size_pixels': full_size_pixels,
            'tile_size': self.grid_size,
            'tile_size_pixels': tile_size_pixels,
            'grid_scale': self.grid_scale,
            'grid_step': self.grid_step,
            'tiles_info': tiles_info,
            'x_min': x_min,
            'y_min': y_min,
        }

        print(f"[bev] Processed {len(tiles_info)} tiles")
        print(f"[bev] Full image shape: {full_rgb.shape}")

        return full_rgb, full_gt, metadata

    def stitch_predictions(self, predictions: List[Tuple[int, int, np.ndarray]], metadata: dict) -> np.ndarray:
        full_size_pixels = metadata['full_size_pixels']
        full_pred = np.zeros((full_size_pixels, full_size_pixels), dtype=np.int64)

        for x_idx, y_idx, pred_tile in predictions:
            tile_info = next(
                (t for t in metadata['tiles_info']
                 if t['x_idx'] == x_idx and t['y_idx'] == y_idx),
                None
            )
            if tile_info is None:
                continue

            x_px = tile_info['x_px']
            y_px = tile_info['y_px']
            tile_h = tile_info['tile_h']
            tile_w = tile_info['tile_w']

            pred_portion = pred_tile[:tile_h, :tile_w]
            full_pred[y_px:y_px+tile_h, x_px:x_px+tile_w] = pred_portion

        return full_pred
