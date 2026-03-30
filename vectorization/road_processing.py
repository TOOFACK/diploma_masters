#!/usr/bin/env python3
"""
Road mask post-processing: fill holes, stitch gaps, and straighten edges.

Uses skimage.morphology.skeletonize instead of pure-Python Zhang-Suen thinning
and vectorized skeleton endpoint detection via cv2.filter2D.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


class RoadMaskProcessor:
    """Encapsulates road mask preprocessing pipeline."""

    def __init__(self, params: Dict):
        self.params = params or {}

    @staticmethod
    def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
        """Fills holes inside road regions using flood fill."""
        h, w = binary_mask.shape[:2]
        flood = binary_mask.copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(binary_mask, flood_inv)
        return filled

    @staticmethod
    def fill_small_holes(binary_mask: np.ndarray, max_area: int) -> np.ndarray:
        """Fills small internal holes by area."""
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return binary_mask
        for idx, cnt in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent == -1:
                continue
            area = cv2.contourArea(cnt)
            if area <= max_area:
                cv2.drawContours(binary_mask, [cnt], -1, 255, -1)
        return binary_mask

    @staticmethod
    def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
        """Skeletonizes binary mask using skimage (C implementation, ~100x faster than Zhang-Suen in Python)."""
        bool_img = binary_mask > 0
        skel_bool = skeletonize(bool_img)
        return (skel_bool.astype(np.uint8)) * 255

    @staticmethod
    def skeleton_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
        """Finds skeleton endpoints using vectorized 3x3 convolution."""
        skel_bin = (skel > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel_bin, cv2.CV_16S, kernel) - skel_bin.astype(np.int16)
        endpoints = np.argwhere((skel_bin > 0) & (neighbor_count == 1))
        return [(int(x), int(y)) for y, x in endpoints]

    @staticmethod
    def estimate_direction(skel: np.ndarray, start: Tuple[int, int], steps: int) -> Optional[np.ndarray]:
        """Estimates direction along skeleton from an endpoint."""
        x, y = start
        prev = None
        current = (x, y)
        h, w = skel.shape
        for _ in range(steps):
            cx, cy = current
            neighbors = []
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if nx == cx and ny == cy:
                        continue
                    if skel[ny, nx] == 0:
                        continue
                    if prev is not None and (nx, ny) == prev:
                        continue
                    neighbors.append((nx, ny))
            if not neighbors:
                break
            nxt = neighbors[0]
            prev = current
            current = nxt
        dx = current[0] - x
        dy = current[1] - y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        return np.array([dx / norm, dy / norm], dtype=np.float32)

    @staticmethod
    def cast_ray_to_road(mask: np.ndarray, start: Tuple[int, int], direction: np.ndarray,
                         max_dist: int, step: float) -> Optional[Tuple[int, int, float]]:
        """Casts a ray and returns the first road pixel hit."""
        h, w = mask.shape
        x0, y0 = float(start[0]), float(start[1])
        dist = step
        while dist <= max_dist:
            x = int(round(x0 + direction[0] * dist))
            y = int(round(y0 + direction[1] * dist))
            if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
                dist += step
                continue
            if mask[y, x] > 0:
                return (x, y, dist)
            dist += step
        return None

    def stitch_road_gaps(self, binary_mask: np.ndarray) -> np.ndarray:
        """Stitches road gaps using skeleton endpoints and ray casting."""
        max_gap = int(self.params.get('max_gap', 25))
        min_gap = int(self.params.get('min_gap', 3))
        ray_step = float(self.params.get('ray_step', 1.0))
        direction_steps = int(self.params.get('direction_steps', 6))
        target_road_count = int(self.params.get('target_road_count', 1))
        max_join_iterations = int(self.params.get('max_join_iterations', 10))
        post_smooth_kernel = int(self.params.get('post_smooth_kernel', 3))

        mask = binary_mask.copy()
        for _ in range(max_join_iterations):
            num_labels, labels = cv2.connectedComponents(mask)
            if num_labels - 1 <= target_road_count:
                break
            skel = self.skeletonize_mask(mask)
            endpoints = self.skeleton_endpoints(skel)
            if not endpoints:
                break
            dist_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            stitched = False
            for x, y in endpoints:
                comp_id = labels[y, x]
                if comp_id == 0:
                    continue
                direction = self.estimate_direction(skel, (x, y), direction_steps)
                if direction is None:
                    continue
                hit = self.cast_ray_to_road(mask, (x, y), direction, max_gap, ray_step)
                if hit is None:
                    continue
                hx, hy, dist = hit
                if dist < min_gap:
                    continue
                if labels[hy, hx] == comp_id:
                    continue
                width = max(1, int(2.0 * dist_map[y, x]))
                thickness = max(1, int(width))
                cv2.line(mask, (x, y), (hx, hy), 255, thickness)
                stitched = True
            if not stitched:
                break
            if post_smooth_kernel > 1:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (post_smooth_kernel, post_smooth_kernel)
                )
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def straighten_road_edges(self, binary_mask: np.ndarray) -> np.ndarray:
        """Straightens road borders while keeping turns with a small step."""
        if not self.params.get('straighten_enabled', True):
            return binary_mask
        straighten_step = float(self.params.get('straighten_step', 3.0))
        angle_tol = float(self.params.get('straighten_angle_tolerance_deg', 20.0))
        min_area = float(self.params.get('straighten_min_area', 50.0))

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(binary_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                cv2.drawContours(out, [cnt], -1, 255, -1)
                continue
            approx = cv2.approxPolyDP(cnt, straighten_step, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) < 3:
                cv2.drawContours(out, [cnt], -1, 255, -1)
                continue
            new_pts = [pts[0]]
            for p in pts[1:]:
                p1 = new_pts[-1]
                dx = p[0] - p1[0]
                dy = p[1] - p1[1]
                ang = abs(math.degrees(math.atan2(dy, dx))) % 180
                if ang <= angle_tol or ang >= (180 - angle_tol):
                    p = [p[0], p1[1]]
                elif abs(ang - 90) <= angle_tol:
                    p = [p1[0], p[1]]
                new_pts.append(p)
            poly = np.array(new_pts, dtype=np.int32)
            cv2.fillPoly(out, [poly], 255)
        return out

    def preprocess(self, binary_mask: np.ndarray,
                   debug_dir: Optional[str] = None,
                   debug_prefix: str = "road") -> np.ndarray:
        """Full road mask preprocessing pipeline with optional debug dumps."""
        base_kernel = int(self.params.get('base_kernel', 3))
        holes_max_area = int(self.params.get('fill_holes_max_area', 200))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (base_kernel, base_kernel))
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        if debug_dir:
            cv2.imwrite(str(Path(debug_dir) / f"{debug_prefix}_01_close.png"), mask)
        mask = self.fill_holes(mask)
        mask = self.fill_small_holes(mask, holes_max_area)
        if debug_dir:
            cv2.imwrite(str(Path(debug_dir) / f"{debug_prefix}_02_holes_filled.png"), mask)
        mask = self.stitch_road_gaps(mask)
        if debug_dir:
            cv2.imwrite(str(Path(debug_dir) / f"{debug_prefix}_03_stitched.png"), mask)
        mask = self.fill_holes(mask)
        mask = self.fill_small_holes(mask, holes_max_area)
        mask = self.straighten_road_edges(mask)
        if debug_dir:
            cv2.imwrite(str(Path(debug_dir) / f"{debug_prefix}_04_straightened.png"), mask)
        return mask
