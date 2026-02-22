#!/usr/bin/env python3
"""
Joint BEV + 3D Segmentation Pipeline.

Per-tile flow:
  grid_points (N, 7)
    ├── BEV branch: project_bev → RGB → SegFormer → bev_logits (14, H, H)
    ├── 3D branch:  prepare_tile → Concerto → logits_3d (N, 13) → project_to_BEV → (14, H, H)
    └── Merge → pred_tile (H, H) → vectorize (optional)

Modes (--mode):
  bev_only  — BEV SegFormer only (baseline)
  3d_only   — 3D Concerto projected to BEV
  joint     — merge BEV + 3D

Merge strategies (--merge-strategy, joint mode only):
  weighted_softmax     — α·softmax(BEV) + (1-α)·softmax(3D)
  confidence_selection — per-pixel pick higher max(softmax)
  3d_priority          — BEV base, override where 3D confidence > threshold
"""

import argparse
import math
import os
import sys
import yaml
import tempfile
import shutil
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# Add parent directory to path (for Concerto imports via concerto_inference)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ply_tile_processor import PLYTileProcessor
from tile_vectorizer import TileVectorizer
from predict_ply import (
    build_model, load_checkpoint, preprocess_image, mask_to_color, save_results
)
from concerto_inference import ConcertoInference
from merge_strategies import (
    project_3d_logits_to_bev,
    merge_weighted_softmax,
    merge_confidence_selection,
    merge_3d_priority,
)


# -------------------------
# BEV tile logits
# -------------------------

@torch.no_grad()
def predict_tile_logits(model, tile_rgb, device="cuda"):
    """
    Get BEV SegFormer logits for a single tile.

    Args:
        model: SegFormer model
        tile_rgb: (H, W, 3) uint8 numpy array
        device: device string

    Returns:
        logits: (C, H, W) numpy float32
    """
    model.eval()
    tile_tensor = preprocess_image(tile_rgb).to(device)
    output = model(tile_tensor)

    if isinstance(output, dict):
        logits = output.get('logits', output.get('out', None))
    elif hasattr(output, 'logits'):
        logits = output.logits
    else:
        logits = output

    if logits.shape[-2:] != tile_tensor.shape[-2:]:
        logits = F.interpolate(
            logits, size=tile_tensor.shape[-2:],
            mode="bilinear", align_corners=False
        )

    return logits[0].detach().cpu().numpy()  # (C, H, W) float32


# -------------------------
# Main processing loop
# -------------------------

def process_tiles_joint(
    processor,
    ply_data,
    bev_model,
    concerto_infer,
    mode,
    merge_strategy,
    merge_alpha,
    merge_threshold,
    device,
    max_tiles=None,
    vectorizer=None,
    output_dir=None,
):
    """
    Process tiles with joint BEV + 3D pipeline.

    Args:
        processor: PLYTileProcessor instance
        ply_data: (N, 7) raw PLY data (global coords)
        bev_model: SegFormer model (or None for 3d_only)
        concerto_infer: ConcertoInference (or None for bev_only)
        mode: 'bev_only', '3d_only', or 'joint'
        merge_strategy: merge strategy name
        merge_alpha: alpha for weighted_softmax
        merge_threshold: threshold for 3d_priority
        device: torch device string
        max_tiles: limit number of tiles (debug mode)
        vectorizer: TileVectorizer (or None)
        output_dir: save per-tile debug images if set

    Returns:
        (full_rgb, full_gt, full_pred, metadata, full_vectorized,
         full_pred_bev, full_pred_3d)
    """
    # Normalize coordinates (same as process_ply_to_bev)
    ply_data = ply_data.copy()
    x_min = ply_data[:, 0].min()
    y_min = ply_data[:, 1].min()
    ply_data[:, 0] -= x_min
    ply_data[:, 1] -= y_min

    x_max = ply_data[:, 0].max()
    y_max = ply_data[:, 1].max()
    full_size = math.ceil(max(x_max, y_max) + processor.grid_size)
    full_size_pixels = int(full_size / processor.grid_scale)
    tile_size_pixels = int(processor.grid_size / processor.grid_scale)

    print(f"[bev] Full size: {full_size}m -> {full_size_pixels}px")
    print(f"[bev] Tile size: {processor.grid_size}m -> {tile_size_pixels}px")

    full_rgb = np.zeros((full_size_pixels, full_size_pixels, 3), dtype=np.uint8)
    full_gt = np.zeros((full_size_pixels, full_size_pixels), dtype=np.int64)

    predictions = []
    predictions_bev = []  # individual BEV predictions (for joint comparison)
    predictions_3d = []   # individual 3D predictions (for joint comparison)
    vectorized_tiles = []

    metadata = {
        'full_size': full_size,
        'full_size_pixels': full_size_pixels,
        'tile_size': processor.grid_size,
        'tile_size_pixels': tile_size_pixels,
        'grid_scale': processor.grid_scale,
        'grid_step': processor.grid_step,
        'tiles_info': [],
        'x_min': x_min,
        'y_min': y_min,
    }

    grid_gen = processor.grid_generator(ply_data, margin=False)
    tile_count = 0

    for x_idx, y_idx, grid_points in grid_gen:
        if max_tiles is not None and tile_count >= max_tiles:
            print(f"[debug] Limited to {max_tiles} tiles")
            break
        tile_count += 1

        tile_id = f"{x_idx}_{y_idx}"
        print(f"\n[tile {tile_count}] {tile_id} ({grid_points.shape[0]} points)")

        # --- BEV projection (always needed for RGB/GT) ---
        alt, rgb, cla = processor.project_bev(grid_points)

        # Place into full image
        x_px = int(x_idx / processor.grid_scale)
        y_px = int(y_idx / processor.grid_scale)
        tile_h = min(tile_size_pixels, full_size_pixels - y_px)
        tile_w = min(tile_size_pixels, full_size_pixels - x_px)

        full_rgb[y_px:y_px+tile_h, x_px:x_px+tile_w] = rgb[:tile_h, :tile_w]
        full_gt[y_px:y_px+tile_h, x_px:x_px+tile_w] = cla[:tile_h, :tile_w]

        tile_info = {
            'x_idx': x_idx, 'y_idx': y_idx,
            'x_px': x_px, 'y_px': y_px,
            'tile_h': tile_h, 'tile_w': tile_w,
        }
        metadata['tiles_info'].append(tile_info)

        # --- BEV branch ---
        bev_logits = None
        if mode in ('bev_only', 'joint'):
            print(f"  [BEV] Running SegFormer...")
            bev_logits = predict_tile_logits(bev_model, rgb, device)  # (14, H, W)

        # --- 3D branch ---
        logits_3d_bev = None
        if mode in ('3d_only', 'joint'):
            print(f"  [3D] Running Concerto ({grid_points.shape[0]} pts)...")
            point_data = concerto_infer.prepare_tile_data(
                grid_points, processor.grid_size
            )
            logits_3d, pred_3d = concerto_infer.infer_tile_logits(point_data)
            logits_3d_bev = project_3d_logits_to_bev(
                grid_points, logits_3d,
                processor.grid_scale, processor.grid_size
            )  # (14, H, W)

        # --- Merge ---
        if mode == 'bev_only':
            pred_tile = bev_logits.argmax(axis=0).astype(np.int64)
        elif mode == '3d_only':
            pred_tile = logits_3d_bev.argmax(axis=0).astype(np.int64)
        elif mode == 'joint':
            if merge_strategy == 'weighted_softmax':
                pred_tile = merge_weighted_softmax(
                    bev_logits, logits_3d_bev, merge_alpha
                )
            elif merge_strategy == 'confidence_selection':
                pred_tile = merge_confidence_selection(
                    bev_logits, logits_3d_bev
                )
            elif merge_strategy == '3d_priority':
                pred_tile = merge_3d_priority(
                    bev_logits, logits_3d_bev, merge_threshold
                )
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

            # Also compute individual predictions for comparison
            pred_bev = bev_logits.argmax(axis=0).astype(np.int64)
            pred_3d_bev = logits_3d_bev.argmax(axis=0).astype(np.int64)
            predictions_bev.append((x_idx, y_idx, pred_bev))
            predictions_3d.append((x_idx, y_idx, pred_3d_bev))

        print(f"  [result] pred classes: {np.unique(pred_tile)}")
        predictions.append((x_idx, y_idx, pred_tile))

        # --- Vectorize ---
        if vectorizer is not None:
            vec_png, vec_json = vectorizer.vectorize_tile(pred_tile, tile_id)
            if vec_png:
                vectorized_tiles.append((x_idx, y_idx, vec_png))

        # --- Debug: save per-tile images ---
        if output_dir is not None:
            tile_dir = os.path.join(output_dir, "tiles", tile_id)
            os.makedirs(tile_dir, exist_ok=True)

            Image.fromarray(rgb, mode="RGB").save(
                os.path.join(tile_dir, "rgb.png"))
            Image.fromarray(mask_to_color(cla, 14), mode="RGB").save(
                os.path.join(tile_dir, "gt_color.png"))
            Image.fromarray(mask_to_color(pred_tile, 14), mode="RGB").save(
                os.path.join(tile_dir, f"pred_{mode}_color.png"))

            if mode == 'joint':
                Image.fromarray(mask_to_color(pred_bev, 14), mode="RGB").save(
                    os.path.join(tile_dir, "pred_bev_color.png"))
                Image.fromarray(mask_to_color(pred_3d_bev, 14), mode="RGB").save(
                    os.path.join(tile_dir, "pred_3d_color.png"))

    # --- Stitch ---
    print("\n[stitch] Stitching predictions...")
    full_pred = processor.stitch_predictions(predictions, metadata)

    # Stitch comparison predictions for joint mode
    full_pred_bev = None
    full_pred_3d = None
    if mode == 'joint' and predictions_bev:
        full_pred_bev = processor.stitch_predictions(predictions_bev, metadata)
        full_pred_3d = processor.stitch_predictions(predictions_3d, metadata)

    # Stitch vectorized tiles
    full_vectorized = None
    if vectorizer is not None and vectorized_tiles:
        print("[stitch] Stitching vectorized tiles...")
        full_vectorized = vectorizer.stitch_vectorized_tiles(
            vectorized_tiles, metadata
        )

    return (full_rgb, full_gt, full_pred, metadata, full_vectorized,
            full_pred_bev, full_pred_3d)


# -------------------------
# Config & Main
# -------------------------

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser("Joint BEV + 3D Segmentation Pipeline")
    parser.add_argument("--input", required=True, help="Path to input PLY file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Mode and merge (per-run)
    parser.add_argument("--mode", choices=["bev_only", "3d_only", "joint"],
                        default="joint", help="Pipeline mode")
    parser.add_argument("--merge-strategy",
                        choices=["weighted_softmax", "confidence_selection",
                                 "3d_priority"],
                        default="weighted_softmax",
                        help="Merge strategy for joint mode")
    parser.add_argument("--merge-alpha", type=float, default=0.5,
                        help="BEV weight for weighted_softmax (default: 0.5)")
    parser.add_argument("--merge-threshold", type=float, default=0.7,
                        help="3D confidence threshold for 3d_priority (default: 0.7)")

    # Overrides for config values (None = use config)
    parser.add_argument("--concerto-ckpt", type=str, default=None,
                        help="Override concerto_ckpt from config")
    parser.add_argument("--concerto-grid-size", type=float, default=None,
                        help="Override concerto_grid_size from config")
    parser.add_argument("--grid-scale", type=float, default=None,
                        help="Override grid_scale from config")
    parser.add_argument("--grid-size", type=float, default=None,
                        help="Override grid_size from config")
    parser.add_argument("--grid-step", type=int, default=None,
                        help="Override grid_step from config")

    # Debug & output (per-run)
    parser.add_argument("--debug-tiles", type=int, default=None,
                        help="Limit number of tiles (debug mode)")
    parser.add_argument("--vectorize", action="store_true",
                        help="Enable vectorization of predictions")
    parser.add_argument("--vectorization-config", type=str, default=None,
                        help="Path to vectorization config YAML")
    parser.add_argument("--vectorization-output-dir", type=str, default=None,
                        help="Directory for vectorization outputs")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Override gpu_ids from config")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA")

    args = parser.parse_args()

    # --- Load config (persistent params) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    if os.path.exists(config_path):
        print(f"[config] Loading: {config_path}")
        config = load_config(config_path)
    else:
        print(f"[warning] Config not found: {config_path}, using defaults")
        config = {}

    # Resolve params: CLI overrides config, config overrides hardcoded defaults
    resume_path = config.get("resume")
    num_labels = config.get("num_labels", 14)
    concerto_ckpt = args.concerto_ckpt or config.get("concerto_ckpt")
    concerto_grid_size = args.concerto_grid_size if args.concerto_grid_size is not None else config.get("concerto_grid_size", 0.05)
    grid_scale = args.grid_scale if args.grid_scale is not None else config.get("grid_scale", 0.05)
    grid_size = args.grid_size if args.grid_size is not None else config.get("grid_size", 25)
    grid_step = args.grid_step if args.grid_step is not None else config.get("grid_step", 25)
    gpu_ids_str = args.gpu_ids if args.gpu_ids is not None else config.get("gpu_ids", "0")
    no_cuda = args.no_cuda or config.get("no_cuda", False)

    # Device setup
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
        torch.cuda.set_device(gpu_ids[0])

    print(f"[config] mode={args.mode}, merge={args.merge_strategy}, "
          f"alpha={args.merge_alpha}, threshold={args.merge_threshold}")
    print(f"[config] grid: scale={grid_scale}, size={grid_size}, step={grid_step}")
    print(f"[device] {device}")

    # --- Load BEV model ---
    bev_model = None
    if args.mode in ('bev_only', 'joint'):
        print(f"[BEV] Building SegFormer with {num_labels} classes...")
        bev_model = build_model(num_labels)
        if resume_path and os.path.exists(resume_path):
            load_checkpoint(bev_model, resume_path)
        else:
            print("[warning] No BEV checkpoint, using random weights")

        if use_cuda and len(gpu_ids_str.split(",")) > 1:
            bev_model = torch.nn.DataParallel(
                bev_model,
                device_ids=[int(x) for x in gpu_ids_str.split(",")]
            )
        bev_model = bev_model.to(device)

    # --- Load Concerto model ---
    concerto_infer = None
    if args.mode in ('3d_only', 'joint'):
        if not concerto_ckpt:
            raise ValueError(
                "Concerto checkpoint required for 3d_only/joint mode. "
                "Set concerto_ckpt in config.yaml or pass --concerto-ckpt"
            )
        concerto_infer = ConcertoInference(
            ckpt_path=concerto_ckpt,
            device=device,
            grid_sample_size=concerto_grid_size,
        )

    # --- Load PLY ---
    processor = PLYTileProcessor(
        grid_scale=grid_scale,
        grid_size=grid_size,
        grid_step=grid_step,
    )
    print(f"[load] Loading PLY: {args.input}")
    ply_data = processor.load_ply(args.input, reformat=True)
    print(f"[load] {ply_data.shape[0]} points loaded")

    # --- Setup vectorizer ---
    vectorizer = None
    temp_dir = None
    if args.vectorize:
        vec_config = args.vectorization_config
        if vec_config is None:
            default_cfg = os.path.join(
                os.path.dirname(__file__), '..', '..',
                'vectorization', "input_format.yaml"
            )
            if os.path.exists(default_cfg):
                vec_config = default_cfg
                print(f"[vectorize] Using default config: {vec_config}")
            else:
                print("[warning] No vectorization config found, disabling")
                args.vectorize = False

        if args.vectorize:
            temp_dir = tempfile.mkdtemp(prefix="joint_vec_")
            vec_output = args.vectorization_output_dir or os.path.join(
                temp_dir, "vectorized"
            )
            try:
                vectorizer = TileVectorizer(
                    vectorization_config=vec_config,
                    output_dir=vec_output,
                    temp_dir=temp_dir,
                    color_space="RGB",
                )
                print("[vectorize] Initialized vectorizer")
            except Exception as e:
                print(f"[warning] Vectorizer init failed: {e}")
                vectorizer = None

    # --- Process tiles ---
    debug_output = args.output_dir if args.debug_tiles else None

    (full_rgb, full_gt, full_pred, metadata,
     full_vectorized, full_pred_bev, full_pred_3d) = process_tiles_joint(
        processor=processor,
        ply_data=ply_data,
        bev_model=bev_model,
        concerto_infer=concerto_infer,
        mode=args.mode,
        merge_strategy=args.merge_strategy,
        merge_alpha=args.merge_alpha,
        merge_threshold=args.merge_threshold,
        device=device,
        max_tiles=args.debug_tiles,
        vectorizer=vectorizer,
        output_dir=debug_output,
    )

    # --- Save results ---
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    suffix = f"_{args.mode}"
    if args.mode == 'joint':
        suffix += f"_{args.merge_strategy}"

    save_results(
        full_rgb, full_gt, full_pred, args.output_dir,
        f"{base_name}{suffix}", full_vectorized
    )

    # Save comparison results for joint mode
    if args.mode == 'joint' and full_pred_bev is not None:
        save_results(
            full_rgb, full_gt, full_pred_bev, args.output_dir,
            f"{base_name}_bev_only"
        )
        save_results(
            full_rgb, full_gt, full_pred_3d, args.output_dir,
            f"{base_name}_3d_only"
        )

    # Cleanup temp directory
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n[done] Results saved to: {args.output_dir}")
    print(f"[info] Mode: {args.mode}, Strategy: {args.merge_strategy}")
    print(f"[info] Full image shape: {full_rgb.shape}")
    print(f"[info] Prediction range: [{full_pred.min()}, {full_pred.max()}]")


if __name__ == "__main__":
    main()
