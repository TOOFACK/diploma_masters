#!/usr/bin/env python3
# predict_tile.py
# Predict segmentation for large images by tiling and stitching
#
# Classes (14 classes as in cityscapes_rgbt.py):
#   0: Unlabeled, 1: Ground, 2: Vegetation, 3: Building, 4: Wall, 5: Bridge,
#   6: Parking, 7: Rail, 8: Traffic, 9: Street, 10: Car, 11: Footpath,
#   12: Bike, 13: Water

import argparse
import os
import sys
import math
import yaml
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modeling.unet import Unet


# -------------------------
# Image tiling functions (similar to point_EDA_31.py grid_generator)
# -------------------------
def generate_tiles(image: np.ndarray, tile_size: int, tile_step: int, margin: bool = False) -> List[Tuple[int, int, np.ndarray]]:
    """
    Generate tiles from large image using sliding window approach.
    
    Args:
        image: (H, W, C) numpy array
        tile_size: size of each tile
        tile_step: step size for sliding window
        margin: whether to include margin tiles
    
    Returns:
        List of (x_idx, y_idx, tile) tuples
    """
    h, w = image.shape[:2]
    margin_val = 1 if margin else 0
    
    tiles = []
    x_max = math.ceil(w)
    x_min = 0
    
    for x_start in range(x_min, x_max + margin_val, tile_step):
        x_end = min(x_start + tile_size, w)
        if x_end <= x_start:
            continue
            
        y_max = math.ceil(h)
        y_min = 0
        
        for y_start in range(y_min, y_max + margin_val, tile_step):
            y_end = min(y_start + tile_size, h)
            if y_end <= y_start:
                continue
            
            # Extract tile
            tile = image[y_start:y_end, x_start:x_end].copy()
            
            # Pad if necessary to reach tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_h = max(0, tile_size - tile.shape[0])
                pad_w = max(0, tile_size - tile.shape[1])
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            tiles.append((x_start, y_start, tile))
    
    return tiles


def stitch_predictions(tiles: List[Tuple[int, int, np.ndarray]], 
                      original_shape: Tuple[int, int],
                      tile_size: int,
                      tile_step: int,
                      overlap_mode: str = "last") -> np.ndarray:
    """
    Stitch tile predictions back into full image.
    
    Args:
        tiles: List of (x_idx, y_idx, prediction) tuples
        original_shape: (H, W) of original image
        tile_size: size of each tile
        tile_step: step size used for tiling
        overlap_mode: "last" (use last prediction) or "majority" (majority voting)
    
    Returns:
        Stitched prediction mask (H, W)
    """
    h, w = original_shape
    
    if overlap_mode == "majority":
        # Use majority voting for overlaps
        # Store all predictions for each pixel
        from collections import defaultdict
        pixel_votes = defaultdict(list)
        
        for x_start, y_start, pred_tile in tiles:
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)
            pred_portion = pred_tile[:tile_h, :tile_w]
            
            for dy in range(tile_h):
                for dx in range(tile_w):
                    y, x = y_start + dy, x_start + dx
                    if 0 <= y < h and 0 <= x < w:
                        pixel_votes[(y, x)].append(pred_portion[dy, dx])
        
        # Majority vote for each pixel
        full_pred = np.zeros((h, w), dtype=np.int64)
        for (y, x), votes in pixel_votes.items():
            # Use most common class
            from collections import Counter
            full_pred[y, x] = Counter(votes).most_common(1)[0][0]
    else:
        # Simple: use last prediction (faster)
        full_pred = np.zeros((h, w), dtype=np.int64)
        for x_start, y_start, pred_tile in tiles:
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)
            pred_portion = pred_tile[:tile_h, :tile_w]
            full_pred[y_start:y_start+tile_h, x_start:x_start+tile_w] = pred_portion
    
    return full_pred


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


def build_model(num_labels: int, n_channels: int = 3):
    """
    Build model. For SensatUrban, we use Unet with 4 channels (RGB + thermal).
    But for inference, we might have only RGB, so default is 3.
    """
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
    image: (H, W, C) numpy array, uint8 [0-255] or float [0-1]
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
        tile: (H, W, C) numpy array
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
    
    # Get prediction
    pred = logits.argmax(dim=1)  # (1, H, W)
    
    # Convert to numpy
    pred_np = pred[0].detach().cpu().numpy().astype(np.int64)
    
    return pred_np


@torch.no_grad()
def predict_large_image(model, image: np.ndarray, tile_size: int, tile_step: int, 
                       device: str = "cuda", batch_size: int = 1, overlap_mode: str = "last") -> np.ndarray:
    """
    Predict segmentation for large image by tiling.
    
    Args:
        model: PyTorch model
        image: (H, W, C) numpy array
        tile_size: size of each tile
        tile_step: step size for sliding window
        device: device to run on
        batch_size: batch size for processing tiles
    
    Returns:
        Full prediction mask (H, W)
    """
    model.eval()
    original_shape = image.shape[:2]
    
    # Generate tiles
    print(f"[tiling] Generating tiles from image shape {original_shape}...")
    tiles = generate_tiles(image, tile_size, tile_step, margin=False)
    print(f"[tiling] Generated {len(tiles)} tiles")
    
    # Process tiles
    predictions = []
    for i, (x_start, y_start, tile) in enumerate(tqdm(tiles, desc="Predicting tiles")):
        pred_tile = predict_tile(model, tile, device)
        predictions.append((x_start, y_start, pred_tile))
    
    # Stitch predictions
    print("[stitching] Stitching predictions...")
    full_pred = stitch_predictions(predictions, original_shape, tile_size, tile_step, 
                                   overlap_mode=overlap_mode)
    
    return full_pred


# -------------------------
# Main
# -------------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser("Predict segmentation for large images by tiling")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output prediction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--tile-size", type=int, default=500, help="Size of each tile")
    parser.add_argument("--tile-step", type=int, default=500, help="Step size for sliding window")
    parser.add_argument("--n-channels", type=int, default=4, help="Number of input channels (3 for RGB, 4 for RGB+T)")
    parser.add_argument("--overlap-mode", type=str, default="last", choices=["last", "majority"], 
                       help="How to handle overlapping regions: 'last' (faster) or 'majority' (more accurate)")
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
    
    # Override with command line args if provided
    if args.config != "config.yaml" and os.path.exists(args.config):
        config = load_config(args.config)
        resume_path = config.get("resume", resume_path)
        num_labels = config.get("num_labels", num_labels)
    
    # Device setup
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    if use_cuda:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        torch.cuda.set_device(gpu_ids[0])
    
    # Load image
    print(f"[load] Loading image from: {args.input}")
    image = np.array(Image.open(args.input))
    if len(image.shape) == 2:
        # Grayscale, convert to RGB or RGB+T based on n_channels
        if args.n_channels == 4:
            # Create dummy thermal channel (zeros or copy of grayscale)
            image = np.stack([image, image, image, image], axis=-1)
        else:
            image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 3:
        # RGB, add thermal channel if needed
        if args.n_channels == 4:
            # Create dummy thermal channel (zeros or grayscale)
            thermal = np.mean(image, axis=2, keepdims=True).astype(image.dtype)
            image = np.concatenate([image, thermal], axis=2)
    elif image.shape[2] == 4:
        # RGBA or RGB+T
        if args.n_channels == 3:
            image = image[:, :, :3]  # Use RGB
        # else keep all 4 channels
    elif image.shape[2] > 4:
        # More channels, take first n_channels
        image = image[:, :, :args.n_channels]
    
    print(f"[load] Image shape: {image.shape}, expected channels: {args.n_channels}")
    
    # Build model
    print(f"[model] Building model with {num_labels} classes, {args.n_channels} input channels...")
    model = build_model(num_labels=num_labels, n_channels=args.n_channels)
    
    # Load checkpoint
    if resume_path is not None and os.path.exists(resume_path):
        load_checkpoint(model, resume_path)
    else:
        print("[warning] No checkpoint provided, using random weights")
    
    # Multi-GPU
    if use_cuda and len(args.gpu_ids.split(",")) > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args.gpu_ids.split(",")])
    
    model = model.to(device)
    
    # Predict
    print(f"[predict] Starting prediction with tile_size={args.tile_size}, tile_step={args.tile_step}...")
    prediction = predict_large_image(
        model=model,
        image=image,
        tile_size=args.tile_size,
        tile_step=args.tile_step,
        device=device,
        overlap_mode=args.overlap_mode
    )
    
    # Save prediction
    print(f"[save] Saving prediction to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    # Save as grayscale image
    pred_image = Image.fromarray(prediction.astype(np.uint8), mode="L")
    pred_image.save(args.output)
    
    print(f"[done] Prediction saved successfully!")
    print(f"[info] Prediction shape: {prediction.shape}")
    print(f"[info] Prediction range: [{prediction.min()}, {prediction.max()}]")
    print(f"[info] Unique classes: {np.unique(prediction)}")


if __name__ == "__main__":
    main()
