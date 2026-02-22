"""
End-to-end inference pipeline for SensatUrban scenes.

Takes a raw PLY file, tiles it, runs Concerto segmentation per tile,
and stitches predictions back into the original point cloud.

Usage:
    python infer_sensaturban_scene.py \
        --input scene.ply \
        --ckpt model_best.pth \
        --tile-size 50 \
        --grid-size 0.05 \
        --estimate-normals \
        --save /output/dir
"""

import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

import concerto
from concerto.transform import Compose

try:
    import flash_attn  # noqa
except Exception:
    flash_attn = None

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

CLASS_COLORS_255 = np.array(
    [
        [128, 128, 128],  # 0  Ground
        [0, 128, 0],      # 1  Vegetation
        [255, 0, 0],      # 2  Building
        [255, 165, 0],    # 3  Wall
        [128, 0, 128],    # 4  Bridge
        [255, 255, 0],    # 5  Parking
        [139, 69, 19],    # 6  Rail
        [64, 64, 64],     # 7  Traffic Road
        [0, 255, 255],    # 8  Street Furniture
        [0, 0, 255],      # 9  Car
        [255, 192, 203],  # 10 Footpath
        [50, 205, 50],    # 11 Bike
        [0, 0, 128],      # 12 Water
    ],
    dtype=np.float32,
)
CLASS_COLORS = CLASS_COLORS_255 / 255.0

# ----------------------------
# Transform config (matches val config)
# ----------------------------
def make_transform_config(grid_size: float):
    return [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_size,
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
# SegHead + checkpoint utils (from infer_sensaturban_tile.py)
# ----------------------------
class SegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.seg_head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.seg_head(x)


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


def upcast_feat_like_demo(point):
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


# ----------------------------
# Tile data structure
# ----------------------------
@dataclass
class TileInfo:
    """One tile extracted from the scene."""
    tile_id: str
    indices: np.ndarray      # int indices into global point array
    coord: np.ndarray        # (M, 3) float32, locally centered
    color: np.ndarray        # (M, 3) float32, 0-255
    normal: np.ndarray       # (M, 3) float32


# ----------------------------
# 1. Load scene PLY
# ----------------------------
def load_scene_ply(path: str) -> dict:
    """Load a raw PLY file using Open3D. Returns coord (N,3), color (N,3) in 0-255."""
    print(f"[scene] loading {path} ...")
    pcd = o3d.io.read_point_cloud(path)
    coord = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        color = np.asarray(pcd.colors, dtype=np.float32)
        # Open3D loads colors in 0-1 range; convert to 0-255
        if color.max() <= 1.0 + 1e-6:
            color = (color * 255.0).astype(np.float32)
    else:
        color = np.zeros_like(coord, dtype=np.float32)

    print(f"[scene] {coord.shape[0]} points, "
          f"XY range: [{coord[:, 0].min():.1f}..{coord[:, 0].max():.1f}] x "
          f"[{coord[:, 1].min():.1f}..{coord[:, 1].max():.1f}]")
    return {"coord": coord, "color": color}


# ----------------------------
# 2. Tile the scene
# ----------------------------
def tile_scene(
    coord: np.ndarray,
    color: np.ndarray,
    tile_size: float,
    tile_step: float,
    min_points: int,
    estimate_normals: bool,
    normal_knn: int,
) -> list:
    """Split scene into tiles on an XY grid. Returns list of TileInfo."""
    x_min = math.floor(coord[:, 0].min())
    x_max = math.ceil(coord[:, 0].max())
    y_min = math.floor(coord[:, 1].min())
    y_max = math.ceil(coord[:, 1].max())

    step = int(tile_step)
    tiles = []

    for x_start in range(x_min, x_max, step):
        x_end = x_start + tile_size
        x_mask = (coord[:, 0] >= x_start) & (coord[:, 0] < x_end)
        if not x_mask.any():
            continue

        for y_start in range(y_min, y_max, step):
            y_end = y_start + tile_size
            mask = x_mask & (coord[:, 1] >= y_start) & (coord[:, 1] < y_end)
            n_pts = int(mask.sum())
            if n_pts < min_points:
                continue

            indices = np.where(mask)[0]
            tile_coord = coord[indices].copy()
            tile_color = color[indices].copy()

            # Local centering: XY to tile center, Z to ground
            tile_coord[:, 0] -= x_start + tile_size / 2.0
            tile_coord[:, 1] -= y_start + tile_size / 2.0
            tile_coord[:, 2] -= tile_coord[:, 2].min()

            # Normals
            if estimate_normals:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(tile_coord.astype(np.float64))
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn)
                )
                tile_normal = np.asarray(pcd.normals, dtype=np.float32)
            else:
                tile_normal = np.zeros_like(tile_coord, dtype=np.float32)

            tile_id = f"tile_{x_start}_{y_start}"
            tiles.append(TileInfo(
                tile_id=tile_id,
                indices=indices,
                coord=tile_coord,
                color=tile_color,
                normal=tile_normal,
            ))

    print(f"[tiling] {len(tiles)} tiles "
          f"(tile_size={tile_size}, step={tile_step}, min_points={min_points})")
    return tiles


# ----------------------------
# 3. Per-tile inference
# ----------------------------
def infer_tile(tile: TileInfo, model, seg_head, transform, device: str) -> np.ndarray:
    """Run inference on a single tile. Returns int32 predictions (M,)."""
    point = {
        "coord": tile.coord.copy(),
        "color": tile.color.copy(),
        "normal": tile.normal.copy(),
    }
    point = transform(point)

    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor) and device == "cuda":
                point[k] = point[k].cuda(non_blocking=True)

        point = model(point)
        point = upcast_feat_like_demo(point)

        logits = seg_head(point.feat)
        pred_ds = logits.argmax(dim=-1)
        pred = pred_ds[point.inverse].cpu().numpy().astype(np.int32)

    return pred


# ----------------------------
# 4. Stitch predictions
# ----------------------------
def stitch_predictions(tiles: list, preds: list, n_total: int) -> np.ndarray:
    """Merge per-tile predictions into a global array. Unassigned points get -1."""
    global_pred = np.full(n_total, -1, dtype=np.int32)
    for tile, pred in zip(tiles, preds):
        global_pred[tile.indices] = pred
    assigned = int((global_pred >= 0).sum())
    print(f"[stitch] {assigned}/{n_total} points assigned "
          f"({assigned / n_total * 100:.1f}%)")
    return global_pred


# ----------------------------
# 5. Output
# ----------------------------
def save_results(name: str, coord: np.ndarray, pred: np.ndarray,
                 color: np.ndarray, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Prediction PLY (class-colored)
    pred_colors = np.zeros((len(pred), 3), dtype=np.float64)
    valid = pred >= 0
    pred_colors[valid] = CLASS_COLORS[pred[valid]]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
    pred_path = os.path.join(outdir, f"{name}_pred.ply")
    o3d.io.write_point_cloud(pred_path, pcd_pred)
    print(f"[saved] {pred_path}")

    # Original PLY
    orig_c = color.copy().astype(np.float64)
    if orig_c.max() > 1.0:
        orig_c /= 255.0
    orig_c = np.clip(orig_c, 0.0, 1.0)
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd_orig.colors = o3d.utility.Vector3dVector(orig_c)
    orig_path = os.path.join(outdir, f"{name}_original.ply")
    o3d.io.write_point_cloud(orig_path, pcd_orig)
    print(f"[saved] {orig_path}")

    # Raw predictions
    npy_path = os.path.join(outdir, f"{name}_pred.npy")
    np.save(npy_path, pred)
    print(f"[saved] {npy_path}")


def visualize(coord: np.ndarray, pred: np.ndarray, color: np.ndarray):
    pred_colors = np.zeros((len(pred), 3), dtype=np.float64)
    valid = pred >= 0
    pred_colors[valid] = CLASS_COLORS[pred[valid]]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

    orig_c = color.copy().astype(np.float64)
    if orig_c.max() > 1.0:
        orig_c /= 255.0
    orig_c = np.clip(orig_c, 0.0, 1.0)
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd_orig.colors = o3d.utility.Vector3dVector(orig_c)

    print("Opening visualization windows...")
    o3d.visualization.draw_geometries(
        [pcd_pred], window_name="Predictions", width=960, height=720
    )
    o3d.visualization.draw_geometries(
        [pcd_orig], window_name="Original", width=960, height=720
    )


def print_class_summary(pred: np.ndarray):
    print("\nClass summary:")
    uniq, counts = np.unique(pred[pred >= 0], return_counts=True)
    for cls, cnt in zip(uniq, counts):
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"?{cls}"
        print(f"  {cls:2d} - {name:20s} {cnt:>10d} pts")
    n_unassigned = int((pred < 0).sum())
    if n_unassigned > 0:
        print(f"  --   {'(unassigned)':20s} {n_unassigned:>10d} pts")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Concerto segmentation on a raw SensatUrban PLY scene"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw PLY file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=float, default=50.0,
                        help="Tile size in meters (default: 50)")
    parser.add_argument("--tile-step", type=float, default=None,
                        help="Tile stride in meters (default: same as tile-size)")
    parser.add_argument("--grid-size", type=float, default=0.05,
                        help="Voxel grid size for GridSample (default: 0.05)")
    parser.add_argument("--estimate-normals", action="store_true",
                        help="Estimate normals per tile with Open3D KNN")
    parser.add_argument("--normal-knn", type=int, default=30,
                        help="KNN neighbours for normal estimation (default: 30)")
    parser.add_argument("--min-points", type=int, default=100,
                        help="Skip tiles with fewer points (default: 100)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output directory (headless); if omitted, opens Open3D windows")
    args = parser.parse_args()

    if args.tile_step is None:
        args.tile_step = args.tile_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    concerto.utils.set_seed(46647087)

    # --- Load model ---
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
    seg_head = load_backbone_and_head(model, args.ckpt, device)
    model.eval()
    seg_head.eval()

    transform = Compose(make_transform_config(args.grid_size))

    # --- Load scene ---
    scene = load_scene_ply(args.input)
    coord = scene["coord"]
    color = scene["color"]

    # --- Tile ---
    tiles = tile_scene(
        coord, color,
        tile_size=args.tile_size,
        tile_step=args.tile_step,
        min_points=args.min_points,
        estimate_normals=args.estimate_normals,
        normal_knn=args.normal_knn,
    )

    if not tiles:
        print("No tiles generated — check tile-size and min-points.")
        return

    # --- Inference per tile ---
    preds = []
    for i, tile in enumerate(tiles):
        print(f"[infer] tile {i + 1}/{len(tiles)}: {tile.tile_id} "
              f"({tile.coord.shape[0]} pts)")
        pred = infer_tile(tile, model, seg_head, transform, device)
        preds.append(pred)

    # --- Stitch ---
    global_pred = stitch_predictions(tiles, preds, len(coord))
    print_class_summary(global_pred)

    # --- Output ---
    name = os.path.splitext(os.path.basename(args.input))[0]
    if args.save:
        save_results(name, coord, global_pred, color, args.save)
    else:
        visualize(coord, global_pred, color)


if __name__ == "__main__":
    main()
