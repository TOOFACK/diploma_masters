#!/usr/bin/env python3
# predict_ply.py
# Predict segmentation for PLY files by creating BEV projections, tiling, and stitching

import argparse
import os
import sys
import yaml
import tempfile
import shutil
import subprocess
from typing import Tuple, Optional, List
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ply_tile_processor import PLYTileProcessor

# Import tile vectorizer
from tile_vectorizer import TileVectorizer

# -------------------------
# Model loading
# -------------------------
def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    """Load checkpoint into model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if len(state) == 0:
                state = ckpt
    else:
        state = ckpt
    
    model_state = model.state_dict()
    model_has_module = next(iter(model_state)).startswith("module.")
    new_state = {}
    
    for k, v in state.items():
        kk = k
        if kk.startswith("module.") and not model_has_module:
            kk = kk[len("module."):]
        if (not kk.startswith("module.")) and model_has_module:
            kk = "module." + kk
        new_state[kk] = v
    
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (first 10) -> {missing[:10]}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (first 10) -> {unexpected[:10]}")


def build_model(num_labels: int):
    """Build Segformer model."""
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Preprocess image for model input.
    image: (H, W, C) numpy array, uint8 [0-255]
    returns: (1, C, H, W) tensor, normalized
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return image_tensor


# -------------------------
# Prediction
# -------------------------
@torch.no_grad()
def predict_tile(model, tile: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Predict segmentation for a single tile.
    
    Args:
        model: PyTorch model
        tile: (H, W, 3) numpy array, RGB uint8
        device: device to run on
    
    Returns:
        Prediction mask (H, W) with class indices
    """
    model.eval()
    
    # Preprocess
    tile_tensor = preprocess_image(tile).to(device)
    
    # Predict
    output = model(tile_tensor)
    
    # Handle different output formats
    if isinstance(output, dict):
        logits = output.get('logits', output.get('out', None))
    elif hasattr(output, 'logits'):
        logits = output.logits
    else:
        logits = output
    
    # Resize if needed
    if logits.shape[-2:] != tile_tensor.shape[-2:]:
        logits = F.interpolate(logits, size=tile_tensor.shape[-2:], mode="bilinear", align_corners=False)
    
    # Get prediction
    pred = logits.argmax(dim=1)  # (1, H, W)
    
    # Convert to numpy
    pred_np = pred[0].detach().cpu().numpy().astype(np.int64)
    
    return pred_np





@torch.no_grad()
def predict_tiles_from_temp(model, metadata: dict, temp_dir: str, device: str = "cuda", 
                           max_tiles: Optional[int] = None,
                           vectorizer: Optional[TileVectorizer] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict segmentation for all tiles from temp directory.
    
    Args:
        model: PyTorch model
        metadata: metadata from PLYTileProcessor
        temp_dir: temporary directory with tiles
        device: device to run on
    
    Returns:
        Full prediction mask (H, W)
    """
    model.eval()
    
    predictions = []
    vectorized_tiles = []
    tiles_info = metadata['tiles_info']
    
    if max_tiles is not None:
        tiles_info = tiles_info[:max_tiles]
        print(f"[debug] Predicting only {len(tiles_info)} tiles")
    
    for tile_info in tqdm(tiles_info, desc="Predicting tiles"):
        # Load RGB tile
        rgb_path = tile_info['rgb_path']
        rgb_tile = np.array(Image.open(rgb_path))
        
        # Predict
        pred_tile = predict_tile(model, rgb_tile, device)
        
        predictions.append((
            tile_info['x_idx'],
            tile_info['y_idx'],
            pred_tile
        ))
        
        # Vectorize if requested
        if vectorizer is not None:
            tile_id = f"{tile_info['x_idx']}_{tile_info['y_idx']}"
            vec_png, vec_json = vectorizer.vectorize_tile(pred_tile, tile_id)
            if vec_png:
                vectorized_tiles.append((
                    tile_info['x_idx'],
                    tile_info['y_idx'],
                    vec_png
                ))
    
    # Stitch predictions
    print("[stitching] Stitching predictions...")
    # Create processor for stitching (we only need the method, not the full initialization)
    processor = PLYTileProcessor(
        grid_scale=metadata['grid_scale'],
        grid_size=metadata['tile_size'],
        grid_step=metadata.get('grid_step', metadata['tile_size'])  # Use tile_size as step if not specified
    )
    full_pred = processor.stitch_predictions(predictions, metadata)
    
    # Stitch vectorized tiles if available
    full_vectorized = None
    if vectorizer is not None and vectorized_tiles:
        print("[stitching] Stitching vectorized tiles...")
        full_vectorized = vectorizer.stitch_vectorized_tiles(vectorized_tiles, metadata)
    
    return full_pred, full_vectorized


# -------------------------
# Visualization
# -------------------------
def mask_to_color(mask: np.ndarray, nclass: int = 14) -> np.ndarray:
    """
    Convert class mask to colored image using SensatUrban color map.
    
    Args:
        mask: (H, W) class labels
        nclass: number of classes
    
    Returns:
        (H, W, 3) RGB colored mask
    """
    # SensatUrban color map (14 classes: 0=unlabeled, 1-13=classes)
    color_map = [
        [0, 0, 0],          # 0: Unlabeled (black)
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
        mask_c = (mask == cid)
        colored[mask_c] = color_map[cid]
    
    return colored


def save_results(full_rgb: np.ndarray, full_gt: np.ndarray, full_pred: np.ndarray,
                 output_dir: str, base_name: str, full_vectorized: Optional[np.ndarray] = None):
    """
    Save results: original RGB, GT, prediction, and overlays.
    
    Args:
        full_rgb: (H, W, 3) RGB image
        full_gt: (H, W) GT class labels
        full_pred: (H, W) prediction class labels
        output_dir: directory to save results
        base_name: base name for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB image
    rgb_path = os.path.join(output_dir, f"{base_name}_rgb.png")
    Image.fromarray(full_rgb, mode="RGB").save(rgb_path)
    print(f"[save] Saved RGB: {rgb_path}")
    
    # Save GT as grayscale
    gt_path = os.path.join(output_dir, f"{base_name}_gt.png")
    Image.fromarray(full_gt.astype(np.uint8), mode="L").save(gt_path)
    print(f"[save] Saved GT: {gt_path}")
    
    # Save prediction as grayscale
    pred_path = os.path.join(output_dir, f"{base_name}_pred.png")
    Image.fromarray(full_pred.astype(np.uint8), mode="L").save(pred_path)
    print(f"[save] Saved prediction: {pred_path}")
    
    # Create colored masks
    gt_colored = mask_to_color(full_gt, nclass=14)
    pred_colored = mask_to_color(full_pred, nclass=14)
    
    # Save colored masks
    gt_color_path = os.path.join(output_dir, f"{base_name}_gt_color.png")
    Image.fromarray(gt_colored, mode="RGB").save(gt_color_path)
    print(f"[save] Saved GT colored: {gt_color_path}")
    
    pred_color_path = os.path.join(output_dir, f"{base_name}_pred_color.png")
    Image.fromarray(pred_colored, mode="RGB").save(pred_color_path)
    print(f"[save] Saved prediction colored: {pred_color_path}")
    
    # Create overlays
    from PIL import Image as PILImage
    
    rgb_img = PILImage.fromarray(full_rgb, mode="RGB")
    gt_colored_img = PILImage.fromarray(gt_colored, mode="RGB")
    pred_colored_img = PILImage.fromarray(pred_colored, mode="RGB")
    
    # Overlay GT on RGB
    overlay_gt = PILImage.blend(rgb_img, gt_colored_img, alpha=0.45)
    overlay_gt_path = os.path.join(output_dir, f"{base_name}_overlay_gt.png")
    overlay_gt.save(overlay_gt_path)
    print(f"[save] Saved GT overlay: {overlay_gt_path}")
    
    # Overlay prediction on RGB
    overlay_pred = PILImage.blend(rgb_img, pred_colored_img, alpha=0.45)
    overlay_pred_path = os.path.join(output_dir, f"{base_name}_overlay_pred.png")
    overlay_pred.save(overlay_pred_path)
    print(f"[save] Saved prediction overlay: {overlay_pred_path}")
    
    # Save vectorized result if available
    if full_vectorized is not None:
        vectorized_path = os.path.join(output_dir, f"{base_name}_vectorized.png")
        Image.fromarray(full_vectorized, mode="RGB").save(vectorized_path)
        print(f"[save] Saved vectorized: {vectorized_path}")
        
        # Overlay vectorized on RGB
        vectorized_img = PILImage.fromarray(full_vectorized, mode="RGB")
        overlay_vectorized = PILImage.blend(rgb_img, vectorized_img, alpha=0.5)
        overlay_vec_path = os.path.join(output_dir, f"{base_name}_overlay_vectorized.png")
        overlay_vectorized.save(overlay_vec_path)
        print(f"[save] Saved vectorized overlay: {overlay_vec_path}")


# -------------------------
# Main
# -------------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser("Predict segmentation for PLY files")
    parser.add_argument("--input", type=str, required=True, help="Path to input PLY file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--grid-scale", type=float, default=0.05, help="BEV grid scale (meters per pixel)")
    parser.add_argument("--grid-size", type=float, default=25, help="Tile size in meters")
    parser.add_argument("--grid-step", type=int, default=25, help="Step size for sliding window in meters")
    parser.add_argument("--temp-dir", type=str, default=None, help="Temporary directory for tiles (auto if None)")
    parser.add_argument("--keep-temp", action="store_true", default=False, 
                       help="Keep temporary files (default: False, temp dir is deleted)")
    parser.add_argument("--debug-tiles", type=int, default=None, help="Debug mode: limit number of tiles to process (e.g., 4)")
    parser.add_argument("--vectorize", action="store_true", help="Enable vectorization of predictions")
    parser.add_argument("--vectorization-config", type=str, default=None, help="Path to vectorization config YAML")
    parser.add_argument("--vectorization-output-dir", type=str, default=None, help="Directory for vectorization outputs")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs to use")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    if os.path.exists(config_path):
        print(f"[config] loading from: {config_path}")
        config = load_config(config_path)
        resume_path = config.get("resume")
        num_labels = config.get("num_labels", 14)
    else:
        print(f"[config] config not found, using defaults")
        resume_path = None
        num_labels = 14
    
    # Device setup
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    if use_cuda:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        torch.cuda.set_device(gpu_ids[0])
    
    # Setup temp directory
    if args.temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ply_tiles_")
        print(f"[temp] Created temp directory: {temp_dir}")
    else:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[temp] Using temp directory: {temp_dir}")
    
    try:
        # Initialize processor
        processor = PLYTileProcessor(
            grid_scale=args.grid_scale,
            grid_size=args.grid_size,
            grid_step=args.grid_step
        )
        
        # Process PLY to BEV tiles
        print(f"[process] Processing PLY file: {args.input}")
        if args.debug_tiles:
            print(f"[debug] DEBUG MODE: Processing only first {args.debug_tiles} tiles")
        full_rgb, full_gt, metadata = processor.process_ply_to_bev(
            args.input, temp_dir, max_tiles=args.debug_tiles
        )
        
        # Build model
        print(f"[model] Building model with {num_labels} classes...")
        model = build_model(num_labels=num_labels)
        
        # Load checkpoint
        if resume_path is not None and os.path.exists(resume_path):
            load_checkpoint(model, resume_path)
        else:
            print("[warning] No checkpoint provided, using random weights")
        
        # Multi-GPU
        if use_cuda and len(args.gpu_ids.split(",")) > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args.gpu_ids.split(",")])
        
        model = model.to(device)
        
        # Setup vectorization
        vectorizer = None
        if args.vectorize:
            vectorization_config = args.vectorization_config
            if vectorization_config is None:
                # Try to find default config
                default_config = os.path.join(os.path.dirname(__file__), '..', '..', 'vectorization', "input_format.yaml")
                if os.path.exists(default_config):
                    vectorization_config = default_config
                    print(f"[vectorize] Using default config: {vectorization_config}")
                else:
                    print("[warning] Vectorization enabled but no config found, disabling vectorization")
                    args.vectorize = False
            
            if args.vectorize:
                vectorization_output_dir = args.vectorization_output_dir
                if vectorization_output_dir is None:
                    vectorization_output_dir = os.path.join(temp_dir, "vectorized")
                    print(f"[vectorize] Using temp directory for vectorization: {vectorization_output_dir}")
                
                try:
                    vectorizer = TileVectorizer(
                        vectorization_config=vectorization_config,
                        output_dir=vectorization_output_dir,
                        temp_dir=temp_dir,
                        color_space="RGB"
                    )
                    print(f"[vectorize] Initialized vectorizer")
                except Exception as e:
                    print(f"[warning] Failed to initialize vectorizer: {e}")
                    vectorizer = None
                    args.vectorize = False
        
        # Predict tiles
        print(f"[predict] Starting prediction...")
        full_pred, full_vectorized = predict_tiles_from_temp(
            model, metadata, temp_dir, device, 
            max_tiles=args.debug_tiles,
            vectorizer=vectorizer
        )
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        save_results(full_rgb, full_gt, full_pred, args.output_dir, base_name, full_vectorized)
        
        print(f"[done] All done! Results saved to: {args.output_dir}")
        print(f"[info] Full image shape: {full_rgb.shape}")
        print(f"[info] GT range: [{full_gt.min()}, {full_gt.max()}]")
        print(f"[info] Prediction range: [{full_pred.min()}, {full_pred.max()}]")
        
    finally:
        # Cleanup temp directory
        # By default, delete temp directory unless --keep-temp is specified
        # If --temp-dir was explicitly provided, only delete if --keep-temp is False
        should_delete = not args.keep_temp
        
        if should_delete:
            if args.temp_dir is None:
                # Auto-created temp directory - always delete
                print(f"[cleanup] Removing auto-created temp directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # User-specified temp directory - ask or delete based on flag
                print(f"[cleanup] Removing temp directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"[cleanup] Keeping temp directory: {temp_dir}")


if __name__ == "__main__":
    main()
