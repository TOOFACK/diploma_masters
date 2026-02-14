#!/usr/bin/env python3
# validate_filtered.py
# Filter validation images by max_iou threshold and specific classes

import argparse
import os
import sys
import yaml
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision

# Add parent directory to path to import dataloaders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataloaders import make_data_loader


# -------------------------
# Cityscapes palette (trainId -> RGB)
# -------------------------
CITYSCAPES_COLORS_19 = {
    0:  (128, 64, 128),   # road
    1:  (244, 35, 232),   # sidewalk
    2:  (70, 70, 70),     # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170, 30),   # traffic light
    7:  (220, 220, 0),    # traffic sign
    8:  (107, 142, 35),   # vegetation
    9:  (152, 251, 152),  # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}


def mask_to_cityscapes_rgb(mask_hw: np.ndarray, nclass: int, ignore_index: int = 255, 
                          classes: Optional[List[int]] = None) -> np.ndarray:
    """
    mask_hw: (H,W) int array with class ids [0..nclass-1] and maybe ignore_index
    classes: if provided, only visualize these classes (others will be black)
    returns: (H,W,3) uint8 colored exactly by Cityscapes palette (trainId colors)
    """
    if mask_hw.ndim != 2:
        raise ValueError(f"mask_hw must be (H,W), got {mask_hw.shape}")

    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # ignore -> black
    out[mask_hw == ignore_index] = (0, 0, 0)

    # paint classes
    for cid in range(nclass):
        # If classes list is provided, only visualize those classes
        if classes is not None and cid not in classes:
            continue
        color = CITYSCAPES_COLORS_19.get(cid, (0, 0, 0))
        out[mask_hw == cid] = color

    return out


# -------------------------
# IoU utils
# -------------------------
def fast_hist(pred: np.ndarray, gt: np.ndarray, nclass: int) -> np.ndarray:
    """
    pred, gt: 1D arrays of same length, values in [0..nclass-1]
    returns: confusion matrix (nclass, nclass)
    """
    k = (gt >= 0) & (gt < nclass)
    return np.bincount(
        nclass * gt[k].astype(np.int64) + pred[k].astype(np.int64),
        minlength=nclass ** 2,
    ).reshape(nclass, nclass)


def per_class_iou_from_hist(hist: np.ndarray, classes: Optional[List[int]] = None) -> np.ndarray:
    """
    Calculate per-class IoU from confusion matrix.
    If classes is provided, only calculate IoU for those classes.
    """
    diag = np.diag(hist).astype(np.float64)
    denom = (hist.sum(1) + hist.sum(0) - diag).astype(np.float64)
    iou = diag / np.maximum(denom, 1.0)

    gt_count = hist.sum(1)
    iou[gt_count == 0] = np.nan
    
    # If specific classes are requested, mask out others
    if classes is not None:
        mask = np.zeros_like(iou, dtype=bool)
        mask[classes] = True
        iou[~mask] = np.nan
    
    return iou


def calculate_miou_for_classes(iou: np.ndarray, classes: Optional[List[int]] = None) -> float:
    """
    Calculate mean IoU considering only specified classes.
    """
    if classes is not None:
        valid_iou = iou[classes]
        valid_iou = valid_iou[~np.isnan(valid_iou)]
        if len(valid_iou) == 0:
            return np.nan
        return float(np.mean(valid_iou))
    else:
        valid_iou = iou[~np.isnan(iou)]
        if len(valid_iou) == 0:
            return np.nan
        return float(np.mean(valid_iou))


# -------------------------
# Checkpoint / model
# -------------------------
def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # common patterns
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            # fallback: maybe already a state_dict-like dict
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if len(state) == 0:
                # last fallback: treat whole ckpt as state
                state = ckpt
    else:
        state = ckpt

    model_state = model.state_dict()
    model_has_module = next(iter(model_state)).startswith("module.")
    new_state = {}

    for k, v in state.items():
        kk = k
        if kk.startswith("module.") and not model_has_module:
            kk = kk[len("module.") :]
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
    """
    ВАЖНО:
    Здесь стоит SegFormer как пример. Если у тебя чекпойнт от Res-Unet,
    замени build_model на создание твоей модели (Res-Unet) из репозитория.
    """
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


# -------------------------
# Image unnormalize (если dataloader нормализует ImageNet-ом)
# -------------------------
def unnormalize_image(
    img: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    img: (3,H,W) tensor, normalized -> returns float [0..1]
    Если у тебя даталоадер НЕ normalizes ImageNet-ом — можно заменить на img.clamp(0,1)
    """
    mean_t = torch.tensor(mean, device=img.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=img.device).view(3, 1, 1)
    x = img * std_t + mean_t
    return x.clamp(0, 1)


def save_example(
    save_dir: str,
    name: str,
    img_chw: torch.Tensor,        # (3,H,W)
    gt_hw: np.ndarray,            # (H,W)
    pred_hw: np.ndarray,          # (H,W)
    nclass: int,
    ignore_index: int,
    save_overlays: bool = True,
    classes: Optional[List[int]] = None,
):
    os.makedirs(save_dir, exist_ok=True)

    # 1) image
    img = img_chw.detach().cpu()
    img = unnormalize_image(img)
    torchvision.utils.save_image(img, os.path.join(save_dir, f"{name}_image.png"))

    # 2) raw masks (ids)
    gt_u8 = gt_hw.astype(np.uint8)
    pred_u8 = pred_hw.astype(np.uint8)
    # Image.fromarray(gt_u8, mode="L").save(os.path.join(save_dir, f"{name}_gt.png"))
    # Image.fromarray(pred_u8, mode="L").save(os.path.join(save_dir, f"{name}_pred.png"))

    # 3) colored masks in EXACT cityscapes colors (only visualize specified classes if provided)
    gt_rgb = mask_to_cityscapes_rgb(gt_hw.astype(np.int64), nclass=nclass, ignore_index=ignore_index, classes=classes)
    pred_rgb = mask_to_cityscapes_rgb(pred_hw.astype(np.int64), nclass=nclass, ignore_index=ignore_index, classes=classes)

    # Image.fromarray(gt_rgb, mode="RGB").save(os.path.join(save_dir, f"{name}_gt_color.png"))
    # Image.fromarray(pred_rgb, mode="RGB").save(os.path.join(save_dir, f"{name}_pred_color.png"))

    if not save_overlays:
        return

    # 4) overlay pred on image
    img_rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_rgb = Image.fromarray(img_rgb, mode="RGB")
    pred_rgb_img = Image.fromarray(pred_rgb, mode="RGB")

    overlay_pred = Image.blend(img_rgb, pred_rgb_img, alpha=0.45)
    overlay_pred.save(os.path.join(save_dir, f"{name}_overlay_pred.png"))

    # 5) overlay gt on image
    gt_rgb_img = Image.fromarray(gt_rgb, mode="RGB")
    overlay_gt = Image.blend(img_rgb, gt_rgb_img, alpha=0.45)
    overlay_gt.save(os.path.join(save_dir, f"{name}_overlay_gt.png"))


# -------------------------
# Validate + filtered saving
# -------------------------
@torch.no_grad()
def validate_filtered(
    model,
    val_loader,
    nclass: int,
    ignore_index: int = 255,
    device: str = "cuda",
    max_iou: float = 0.5,
    classes: Optional[List[int]] = None,
    max_samples: int = 100,
    save_dir: str = "filtered_val_samples",
    save_overlays: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Validate and save images with mIoU <= max_iou.
    IoU is calculated only for specified classes if provided.
    """
    model.eval()
    hist = np.zeros((nclass, nclass), dtype=np.int64)

    filtered_samples = []  # list of (score, uid, payload)
    uid = 0

    print(f"[filter] max_iou threshold: {max_iou}")
    if classes is not None:
        print(f"[filter] classes for IoU calculation: {classes}")
    else:
        print(f"[filter] using all classes for IoU calculation")

    for bidx, sample in enumerate(tqdm(val_loader, desc="val")):
        image = sample["image"].to(device, non_blocking=True)          # (B,3,H,W)
        target = sample["label"].to(device, non_blocking=True).long()  # (B,H,W)

        out = model(image).logits  # (B,C,h,w)
        if out.shape[-2:] != target.shape[-2:]:
            out = F.interpolate(out, size=target.shape[-2:], mode="bilinear", align_corners=False)

        pred = out.argmax(dim=1)  # (B,H,W)

        # global confusion
        valid = target != ignore_index
        pred_np = pred[valid].detach().cpu().numpy()
        tgt_np = target[valid].detach().cpu().numpy()
        hist += fast_hist(pred_np.reshape(-1), tgt_np.reshape(-1), nclass)

        # Filter images by IoU threshold
        B = pred.shape[0]
        for i in range(B):
            gt_i = target[i]
            pr_i = pred[i]
            v_i = gt_i != ignore_index
            if v_i.sum().item() == 0:
                continue

            gt_flat = gt_i[v_i].detach().cpu().numpy().astype(np.int64)
            pr_flat = pr_i[v_i].detach().cpu().numpy().astype(np.int64)

            h_i = fast_hist(pr_flat.reshape(-1), gt_flat.reshape(-1), nclass)
            iou_i = per_class_iou_from_hist(h_i, classes=classes)
            score = calculate_miou_for_classes(iou_i, classes=classes)

            # Skip if score is NaN or above threshold
            if np.isnan(score) or score > max_iou:
                continue

            payload = {
                "img": image[i].detach().cpu(),  # CHW
                "gt": gt_i.detach().cpu().numpy(),
                "pred": pr_i.detach().cpu().numpy(),
                "bidx": bidx,
                "i": i,
                "iou_per_class": iou_i.copy(),
            }

            filtered_samples.append((score, uid, payload))
            uid += 1

    iou = per_class_iou_from_hist(hist, classes=classes)
    miou = calculate_miou_for_classes(iou, classes=classes)

    # Save filtered samples
    if len(filtered_samples) > 0:
        os.makedirs(save_dir, exist_ok=True)
        # Sort by IoU in descending order (starting from max_iou, going down)
        # This gives us top samples with highest IoU (up to max_iou threshold)
        filtered_sorted = sorted(filtered_samples, key=lambda x: x[0], reverse=True)
        
        # Take top max_samples starting from max_iou (if max_samples > 0)
        if max_samples > 0:
            filtered_sorted = filtered_sorted[:max_samples]
            print(f"[filter] taking top {len(filtered_sorted)} samples (max_samples={max_samples})")

        for rank, (score, _uid, p) in enumerate(filtered_sorted, start=1):
            name = f"rank{rank:03d}_miou{score*100:.2f}_b{p['bidx']:04d}_i{p['i']:02d}"
            save_example(
                save_dir=save_dir,
                name=name,
                img_chw=p["img"],
                gt_hw=p["gt"],
                pred_hw=p["pred"],
                nclass=nclass,
                ignore_index=ignore_index,
                save_overlays=save_overlays,
                classes=classes,
            )

        print(f"[save] saved {len(filtered_sorted)} filtered examples to: {save_dir}")
        if len(filtered_sorted) > 0:
            print(f"[save] IoU range: {filtered_sorted[0][0]*100:.2f}% (best) - {filtered_sorted[-1][0]*100:.2f}% (worst)")
    else:
        print(f"[save] no samples found with mIoU <= {max_iou}")

    return iou, miou, hist


# -------------------------
# Main
# -------------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser("Validate IoU/mIoU + save filtered examples by max_iou threshold")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to config.yaml file")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Directory containing config.yaml (default: script directory)")
    
    args = parser.parse_args()

    # Determine config path
    if args.config_dir:
        config_path = os.path.join(args.config_dir, "config.yaml")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"[config] loading from: {config_path}")
    config = load_config(config_path)

    # Extract parameters from config
    max_iou = config.get("max_iou", 0.5)
    classes = config.get("classes")
    if classes is not None and len(classes) == 0:
        classes = None
    max_samples = config.get("max_samples", 100)
    
    use_cuda = (not config.get("no_cuda", False)) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        gpu_ids = [int(x) for x in str(config.get("gpu_ids", "0")).split(",")]
        torch.cuda.set_device(gpu_ids[0])

    # Create args-like object for dataloader
    class Args:
        pass
    args_obj = Args()
    args_obj.dataset = config.get("dataset", "cityscapes")
    args_obj.batch_size = config.get("batch_size", 4)
    args_obj.base_size = config.get("base_size", 500)
    args_obj.crop_size = config.get("crop_size", 500)
    args_obj.num_labels = config.get("num_labels", 19)
    args_obj.ignore_index = config.get("ignore_index", 255)

    kwargs = {"num_workers": config.get("workers", 4), "pin_memory": use_cuda}
    _, val_loader, nclass = make_data_loader(args_obj, **kwargs)

    if nclass is None:
        nclass = args_obj.num_labels
    else:
        args_obj.num_labels = nclass

    model = build_model(num_labels=args_obj.num_labels)

    resume_path = config.get("resume")
    if resume_path is not None:
        load_checkpoint(model, resume_path)

    if use_cuda and len(str(config.get("gpu_ids", "0")).split(",")) > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in str(config.get("gpu_ids", "0")).split(",")])

    model = model.to(device)

    # GT statistics
    ignore = args_obj.ignore_index
    counts = np.zeros(nclass, dtype=np.int64)

    for sample in val_loader:
        y = sample["label"]
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        y = y.reshape(-1)
        if np.issubdtype(y.dtype, np.floating):
            y = np.rint(y).astype(np.int64)
        else:
            y = y.astype(np.int64)

        y = y[y != ignore]
        y = y[(y >= 0) & (y < nclass)]
        counts += np.bincount(y, minlength=nclass)

    print("GT pixel counts:", counts)
    denom = max(int(counts.sum()), 1)
    print("GT %:", np.round(counts / denom * 100, 3))

    iou, miou, _hist = validate_filtered(
        model=model,
        val_loader=val_loader,
        nclass=args_obj.num_labels,
        ignore_index=args_obj.ignore_index,
        device=device,
        max_iou=max_iou,
        classes=classes,
        max_samples=max_samples,
        save_dir=config.get("save_dir", "filtered_val_samples"),
        save_overlays=config.get("save_overlays", True),
    )

    print("\n=== IoU per class ===")
    for cid, v in enumerate(iou):
        if np.isnan(v):
            print(f"class {cid:02d}: IoU=nan (no GT pixels)")
        else:
            print(f"class {cid:02d}: IoU={v*100:.2f}%")
    print(f"\nmIoU: {miou*100:.2f}%")


if __name__ == "__main__":
    main()
