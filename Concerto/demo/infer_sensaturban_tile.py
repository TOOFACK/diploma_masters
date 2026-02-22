"""
Inference script for SensatUrban tiles using Concerto model.
Loads a tile (directory with .npy files), runs segmentation,
and displays two Open3D windows: predictions and original colors.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

import concerto
from concerto.transform import Compose

try:
    import flash_attn  # noqa
except Exception:
    flash_attn = None

import open3d as o3d

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# SensatUrban 13-class meta
# ----------------------------
NUM_CLASSES = 13
CLASS_NAMES = [
    "Ground",
    "Vegetation",
    "Building",
    "Wall",
    "Bridge",
    "Parking",
    "Rail",
    "Traffic Road",
    "Street Furniture",
    "Car",
    "Footpath",
    "Bike",
    "Water",
]

# Distinct colors for 13 classes (RGB 0..255)
CLASS_COLORS_255 = np.array(
    [
        [128, 128, 128],  # 0  Ground        - gray
        [0, 128, 0],      # 1  Vegetation    - dark green
        [255, 0, 0],      # 2  Building      - red
        [255, 165, 0],    # 3  Wall          - orange
        [128, 0, 128],    # 4  Bridge        - purple
        [255, 255, 0],    # 5  Parking       - yellow
        [139, 69, 19],    # 6  Rail          - brown
        [64, 64, 64],     # 7  Traffic Road  - dark gray
        [0, 255, 255],    # 8  Street Furn.  - cyan
        [0, 0, 255],      # 9  Car           - blue
        [255, 192, 203],  # 10 Footpath      - pink
        [50, 205, 50],    # 11 Bike          - lime green
        [0, 0, 128],      # 12 Water         - navy
    ],
    dtype=np.float32,
)
CLASS_COLORS = CLASS_COLORS_255 / 255.0

# ----------------------------
# Transform config (matches val config from training)
# ----------------------------
TRANSFORM_CONFIG = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "color", "inverse"),
        feat_keys=("coord", "color", "normal"),
    ),
]


# ----------------------------
# SegHead
# ----------------------------
class SegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.seg_head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.seg_head(x)


# ----------------------------
# Checkpoint utils
# ----------------------------
def extract_state_dict(ckpt: dict) -> dict:
    for k in ["state_dict", "model", "net", "module"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def remap_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("backbone."):
            kk = kk[len("backbone."):]
        if kk.startswith("e."):
            kk = kk[len("e."):]
        if kk.startswith("d."):
            kk = kk[len("d."):]
        out[kk] = v
    return out


def split_backbone_and_head(sd: dict):
    head = {}
    backbone = {}
    for k, v in sd.items():
        if k.startswith("seg_head."):
            head[k[len("seg_head."):]] = v
        else:
            backbone[k] = v
    return backbone, head


def load_backbone_and_head(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = remap_keys(extract_state_dict(ckpt))
    sd_backbone, sd_head = split_backbone_and_head(sd)

    incompatible = model.load_state_dict(sd_backbone, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(f"[backbone] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:10] =", missing[:10])
    if unexpected:
        print("  unexpected[:10] =", unexpected[:10])

    if "weight" not in sd_head:
        raise RuntimeError("No seg_head.weight found in checkpoint.")

    num_classes, in_dim = sd_head["weight"].shape
    print(f"[seg_head] detected in_dim={in_dim}, num_classes={num_classes}")

    seg_head = SegHead(in_dim=in_dim, num_classes=num_classes).to(device)
    seg_head.seg_head.weight.data.copy_(sd_head["weight"].to(device))
    if "bias" in sd_head:
        seg_head.seg_head.bias.data.copy_(sd_head["bias"].to(device))
    return seg_head


# ----------------------------
# Tile loading
# ----------------------------
def load_tile(tile_path: str) -> dict:
    """Load a SensatUrban tile from a directory containing .npy files."""
    if not os.path.isdir(tile_path):
        raise ValueError(f"Tile path must be a directory: {tile_path}")

    coord = np.load(os.path.join(tile_path, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(tile_path, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(tile_path, "normal.npy")).astype(np.float32)

    print(f"[tile] loaded {coord.shape[0]} points from {tile_path}")
    return {
        "coord": coord,
        "color": color,
        "normal": normal,
    }


def upcast_feat_like_demo(point):
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


# ----------------------------
# Visualization
# ----------------------------
def save_results(coord, pred, original_color, outdir):
    """Save prediction and original PLY files for later visualization."""
    os.makedirs(outdir, exist_ok=True)

    # Prediction PLY
    pred_colors = CLASS_COLORS[pred]
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord)
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
    pred_path = os.path.join(outdir, "pred.ply")
    o3d.io.write_point_cloud(pred_path, pcd_pred)

    # Original PLY
    orig_c = original_color.copy()
    if orig_c.max() > 1.0:
        orig_c = orig_c / 255.0
    orig_c = np.clip(orig_c, 0.0, 1.0)
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(coord)
    pcd_orig.colors = o3d.utility.Vector3dVector(orig_c)
    orig_path = os.path.join(outdir, "original.ply")
    o3d.io.write_point_cloud(orig_path, pcd_orig)

    # Save raw predictions
    npy_path = os.path.join(outdir, "pred.npy")
    np.save(npy_path, pred)

    print(f"[saved] {pred_path}")
    print(f"[saved] {orig_path}")
    print(f"[saved] {npy_path}")


def visualize_side_by_side(coord, pred, original_color):
    """Show two Open3D windows: predictions and original tile colors."""
    pred_colors = CLASS_COLORS[pred]
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord)
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

    orig_c = original_color.copy()
    if orig_c.max() > 1.0:
        orig_c = orig_c / 255.0
    orig_c = np.clip(orig_c, 0.0, 1.0)

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(coord)
    pcd_orig.colors = o3d.utility.Vector3dVector(orig_c)

    print("Opening visualization windows...")
    print("  Window 1: Predictions (colored by class)")
    print("  Window 2: Original tile (RGB)")

    o3d.visualization.draw_geometries(
        [pcd_pred], window_name="Predictions", width=960, height=720
    )
    o3d.visualization.draw_geometries(
        [pcd_orig], window_name="Original Tile", width=960, height=720
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Concerto segmentation on a SensatUrban tile"
    )
    parser.add_argument(
        "--tile",
        type=str,
        required=True,
        help="Path to tile directory (containing coord.npy, color.npy, normal.npy)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="data/weights/concerto_exp1_sensaturban/model_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.05,
        help="Voxel grid size for GridSample (default: 0.05 from config)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save result PLY files (for headless servers)",
    )
    args = parser.parse_args()

    concerto.utils.set_seed(46647087)

    # 1) Load model
    print("Loading Concerto model...")
    if flash_attn is not None:
        model = concerto.load(
            "concerto_large_outdoor", repo_id="Pointcept/Concerto"
        ).to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)], enable_flash=False
        )
        model = concerto.load(
            "concerto_large_outdoor",
            repo_id="Pointcept/Concerto",
            custom_config=custom_config,
        ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 2) Load seg head from checkpoint
    seg_head = load_backbone_and_head(model, args.ckpt, device)

    model.eval()
    seg_head.eval()

    # 3) Load tile data
    point = load_tile(args.tile)
    original_coord = point["coord"].copy()
    original_color = point["color"].copy()

    # 4) Apply transform
    transform_cfg = TRANSFORM_CONFIG.copy()
    for t in transform_cfg:
        if t.get("type") == "GridSample":
            t["grid_size"] = float(args.grid_size)

    transform = Compose(transform_cfg)
    point = transform(point)

    # 5) Inference
    print("Running inference...")
    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor) and device == "cuda":
                point[k] = point[k].cuda(non_blocking=True)

        point = model(point)
        point = upcast_feat_like_demo(point)

        logits = seg_head(point.feat)
        pred_ds = logits.argmax(dim=-1)
        pred = pred_ds[point.inverse].cpu().numpy().astype(np.int32)

    print(f"Segmentation done. N={pred.shape[0]}")

    # Print predicted classes
    uniq = np.unique(pred)
    print("Predicted classes:")
    for c in uniq:
        if 0 <= c < len(CLASS_NAMES):
            count = int((pred == c).sum())
            print(f"  {c:2d} - {CLASS_NAMES[c]} ({count} pts)")

    # 6) Save and/or visualize
    if args.save:
        save_results(original_coord, pred, original_color, args.save)
    else:
        visualize_side_by_side(original_coord, pred, original_color)


if __name__ == "__main__":
    main()
