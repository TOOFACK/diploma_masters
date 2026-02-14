#!/usr/bin/env python3
# tile_vectorizer.py
# Class for vectorizing segmentation tiles

import os
import sys
import subprocess
import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class TileVectorizer:
    """
    Class for vectorizing segmentation prediction tiles.
    Uses vectorize_mask.py script for processing.
    """

    def __init__(
        self,
        vectorization_config: str,
        output_dir: str,
        temp_dir: str,
        *,
        color_space: str = "RGB",          # <-- FIX: default to RGB
        white_threshold: int = 250         # threshold for "white" pixels during stitching
    ):
        """
        Args:
            vectorization_config: path to vectorization config YAML
            output_dir: directory for vectorization outputs
            temp_dir: temporary directory for intermediate files
            color_space: 'RGB' or 'BGR' expected by vectorize_mask.py + YAML colors
            white_threshold: pixels >= this value treated as white background
        """
        self.vectorization_config = vectorization_config
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.color_space = color_space.upper()
        self.white_threshold = int(white_threshold)

        vectorization_path = os.path.join(os.path.dirname(__file__), '..', '..', 'vectorization')
        self.vectorize_script = os.path.join(vectorization_path, "vectorize_mask.py")

        if not os.path.exists(self.vectorize_script):
            raise FileNotFoundError(f"Vectorization script not found: {self.vectorize_script}")

        if not os.path.exists(self.vectorization_config):
            raise FileNotFoundError(f"Vectorization config not found: {self.vectorization_config}")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.color_space not in ("RGB", "BGR"):
            raise ValueError("color_space must be 'RGB' or 'BGR'")

    def mask_to_color(self, mask: np.ndarray, nclass: int = 14) -> np.ndarray:
        """(H, W) -> (H, W, 3) RGB colored mask"""
        color_map = [
            [0, 0, 0],          # 0: Unlabeled
            [255, 248, 220],    # 1: Ground
            [220, 220, 220],    # 2: Vegetation
            [139, 71, 38],      # 3: Building
            [238, 197, 145],    # 4: Wall
            [70, 130, 180],     # 5: Bridge
            [179, 238, 58],     # 6: Parking
            [110, 139, 61],     # 7: Rail
            [105, 105, 105],    # 8: Traffic
            [0, 0, 128],        # 9: Street
            [205, 92, 92],      # 10: Car
            [244, 164, 96],     # 11: Footpath
            [147, 112, 219],    # 12: Bike
            [255, 228, 225],    # 13: Water
        ]

        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cid in range(min(nclass, len(color_map))):
            colored[mask == cid] = color_map[cid]
        return colored

    def _save_temp_mask(self, rgb_mask: np.ndarray, path: str) -> None:
        """
        Save temp mask in the chosen color_space.
        - If RGB: write using PIL in RGB exactly (no BGR confusion).
        - If BGR: write using cv2 in BGR.
        """
        if self.color_space == "RGB":
            Image.fromarray(rgb_mask, mode="RGB").save(path)
        else:
            bgr = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)

    def vectorize_tile(self, pred_tile: np.ndarray, tile_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Vectorize a single tile prediction.
        Returns (out_png, out_json) or (None, None).
        """
        # quick skip: empty prediction (all unlabeled)
        if np.all(pred_tile == 0):
            return None, None

        pred_colored_rgb = self.mask_to_color(pred_tile, nclass=14)

        temp_mask_path = os.path.join(self.temp_dir, f"pred_colored_{tile_id}.png")
        self._save_temp_mask(pred_colored_rgb, temp_mask_path)

        out_png = os.path.join(self.output_dir, f"vectorized_{tile_id}.png")
        out_json = os.path.join(self.output_dir, f"vectorized_{tile_id}.json")

        cmd = [
            sys.executable, self.vectorize_script,
            "--mask", temp_mask_path,
            "--config", self.vectorization_config,
            "--out_json", out_json,
            "--out_png", out_png,
            "--color_space", self.color_space,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # If script succeeded but produced empty/white output, still return paths;
            # stitching will ignore pure-white.
            return out_png, out_json
        except subprocess.CalledProcessError as e:
            # print full debug
            print(f"[warning] Vectorization failed for tile {tile_id}")
            if e.stdout:
                print("[vectorize stdout]\n", e.stdout)
            if e.stderr:
                print("[vectorize stderr]\n", e.stderr)
            return None, None

    def stitch_vectorized_tiles(self, vectorized_tiles: List[Tuple[int, int, str]], metadata: dict) -> np.ndarray:
        """
        Stitch vectorized tile images back into full image.
        """
        full_size_pixels = metadata['full_size_pixels']
        full_vectorized = np.full((full_size_pixels, full_size_pixels, 3), 255, dtype=np.uint8)

        thr = self.white_threshold

        for x_idx, y_idx, vec_path in vectorized_tiles:
            if vec_path is None or not os.path.exists(vec_path):
                continue

            tile_info = next(
                (t for t in metadata['tiles_info'] if t['x_idx'] == x_idx and t['y_idx'] == y_idx),
                None
            )
            if tile_info is None:
                continue

            x_px = tile_info['x_px']
            y_px = tile_info['y_px']
            tile_h = tile_info['tile_h']
            tile_w = tile_info['tile_w']

            vec_tile_bgr = cv2.imread(vec_path)  # BGR
            if vec_tile_bgr is None:
                continue

            vec_tile_rgb = cv2.cvtColor(vec_tile_bgr, cv2.COLOR_BGR2RGB)
            vec_portion = vec_tile_rgb[:tile_h, :tile_w]

            # non-white mask (robust threshold)
            mask = np.any(vec_portion < thr, axis=2)

            roi = full_vectorized[y_px:y_px+tile_h, x_px:x_px+tile_w]
            roi[mask] = vec_portion[mask]
            full_vectorized[y_px:y_px+tile_h, x_px:x_px+tile_w] = roi

        return full_vectorized
