# demo/kitti_infer_vis.py
import os
import argparse
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# SemanticKITTI 19-class meta
# ----------------------------
KITTI_VALID_CLASS_IDS = tuple(range(19))
KITTI_CLASS_LABELS = (
    "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist",
    "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk",
    "terrain", "pole", "traffic-sign",
)

KITTI_COLOR_MAP = {
    0: (255.0, 0.0, 0.0),
    1: (0.0, 255.0, 0.0),
    2: (0.0, 0.0, 255.0),
    3: (255.0, 255.0, 0.0),
    4: (255.0, 0.0, 255.0),
    5: (0.0, 255.0, 255.0),
    6: (255.0, 128.0, 0.0),
    7: (128.0, 0.0, 255.0),
    8: (128.0, 128.0, 128.0),
    9: (255.0, 192.0, 203.0),
    10: (0.0, 128.0, 128.0),
    11: (255.0, 215.0, 0.0),
    12: (70.0, 130.0, 180.0),
    13: (165.0, 42.0, 42.0),
    14: (50.0, 205.0, 50.0),
    15: (255.0, 99.0, 71.0),
    16: (0.0, 100.0, 0.0),
    17: (211.0, 211.0, 211.0),
    18: (255.0, 255.0, 255.0),
}
CLASS_COLOR = np.array([KITTI_COLOR_MAP[i] for i in KITTI_VALID_CLASS_IDS], dtype=np.float32) / 255.0


# ----------------------------

# ----------------------------
# TRANSFORM_CONFIG = [
#     dict(type="RandomScale", scale=[1, 1]),
#     dict(
#         type="GridSample",
#         grid_size=0.06,
#         hash_type="fnv",
#         mode="train",
#         return_grid_coord=True,
#         return_inverse=True,
#     ),
#     dict(type="CenterShift", apply_z=False),
#     dict(type="NormalizeColor"),
#     dict(type="ToTensor"),
#     dict(
#         type="Collect",
#         keys=("coord", "grid_coord", "color", "inverse"),
#         feat_keys=("coord", "color", "normal"),
#     ),
# ]

TRANSFORM_CONFIG = [
    # dict(type="RandomScale", scale=[0.2, 0.2]),
    dict(type="RandomScale", scale=[1, 1]),

    dict(
        type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    # dict(type="CenterShift", apply_z=False),
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
            head[k[len("seg_head."):]] = v  # weight/bias
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
    print("missing[:10] =", missing[:10])
    print("unexpected[:10] =", unexpected[:10])

    if "weight" not in sd_head:
        raise RuntimeError("No seg_head.weight found in checkpoint. ")

    num_classes, in_dim = sd_head["weight"].shape
    print(f"[seg_head] detected in_dim={in_dim}, num_classes={num_classes}")

    seg_head = SegHead(in_dim=in_dim, num_classes=num_classes).to(device)
    seg_head.seg_head.weight.data.copy_(sd_head["weight"].to(device))
    if "bias" in sd_head:
        seg_head.seg_head.bias.data.copy_(sd_head["bias"].to(device))
    return seg_head


# ----------------------------
# Data loader for SemanticKITTI .bin
# ----------------------------
def load_kitti_bin(path: str):
    # SemanticKITTI velodyne: float32 [x, y, z, intensity]
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    coord = pts[:, :3]
    intensity = pts[:, 3:4]

 
    inten = intensity.copy()
    if inten.size > 0:
        mn = float(inten.min())
        mx = float(inten.max())
        denom = max(mx - mn, 1e-6)
        inten = (inten - mn) / denom
    color = np.repeat(inten, 3, axis=1).astype(np.float32)

    normal = np.zeros_like(coord, dtype=np.float32)
    return {
        "coord": coord.astype(np.float32),
        "color": color,
        "normal": normal,
    }


def load_point_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_kitti_bin(path)
    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)
        coord = np.asarray(pcd.points, dtype=np.float32)
        # color = np.asarray(pcd.colors, dtype=np.float32)
        color = np.zeros((coord.shape[0], 3), dtype=np.float32)
        # normal = np.asarray(pcd.normals, dtype=np.float32)
        normal = np.zeros_like(coord, dtype=np.float32)


        if color.size == 0:
            print("AAA")
            color = np.zeros((coord.shape[0], 3), dtype=np.float32)
        if normal.size == 0:
            print("AAAA")
            normal = np.zeros_like(coord, dtype=np.float32)

        if coord.size > 0:
            finite_mask = np.isfinite(coord).all(axis=1)
            coord = coord[finite_mask]
            color = color[finite_mask]
            normal = normal[finite_mask]

        return {
            "coord": coord,
            "color": color,
            "normal": normal,
        }
    raise ValueError(f"Unsupported input extension: {ext}")


def upcast_feat_like_demo(point):

    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def visualize_and_save(coord, pred, outdir, show: bool, prefix: str = "pred"):
    os.makedirs(outdir, exist_ok=True)

    colors = CLASS_COLOR[pred]  # (N,3) in 0..1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ply_path = os.path.join(outdir, f"{prefix}.ply")
    npy_path = os.path.join(outdir, f"{prefix}.npy")
    o3d.io.write_point_cloud(ply_path, pcd)
    np.save(npy_path, pred)

    print(f"[saved] {ply_path}")
    print(f"[saved] {npy_path}")
    if show:
        o3d.visualization.draw_geometries([pcd])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model_best.pth (contains seg_head.*)")
    parser.add_argument("--input", type=str, required=True, help="SemanticKITTI .bin path")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--grid_size", type=float, default=0.05)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--wo_color", action="store_true")
    parser.add_argument("--wo_normal", action="store_true")

    args = parser.parse_args()

    concerto.utils.set_seed(46647087)


    if flash_attn is not None:
        model = concerto.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").to(device)
    else:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
        model = concerto.load("concerto_large_outdoor", repo_id="Pointcept/Concerto", custom_config=custom_config).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2) load seg head weights from your ckpt + load backbone weights into this model
    seg_head = load_backbone_and_head(model, args.ckpt, device)

    model.eval()
    seg_head.eval()

    # 3) load data
    point = load_point_file(args.input)
    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"], dtype=np.float32)
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"], dtype=np.float32)

    original_coord = point["coord"].copy()

    # 4) transform
    for t in TRANSFORM_CONFIG:
        if t.get("type") == "GridSample":
            t["grid_size"] = float(args.grid_size)

    transform = Compose(TRANSFORM_CONFIG)
    point = transform(point)

    # 5) inference
    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor) and device == "cuda":
                point[k] = point[k].cuda(non_blocking=True)

        point = model(point)
        point = upcast_feat_like_demo(point)

        logits = seg_head(point.feat)            # (N_ds, 19)
        pred_ds = logits.argmax(dim=-1)          # (N_ds,)
        pred = pred_ds[point.inverse].cpu().numpy().astype(np.int32)  # (N_orig,)
        coord_ds = point.coord.cpu().numpy()

    print(f"Segmentation done. N={pred.shape[0]}")
    uniq = np.unique(pred)
    print("Predicted classes:", uniq.tolist())
    # optional: print labels
    for c in uniq[:10]:
        if 0 <= int(c) < len(KITTI_CLASS_LABELS):
            print(f"  {int(c)} -> {KITTI_CLASS_LABELS[int(c)]}")

    # 6) visualize/save
    visualize_and_save(coord_ds, pred_ds.cpu().numpy().astype(np.int32), args.outdir, args.show, prefix="pred_ds")
    visualize_and_save(original_coord, pred, args.outdir, args.show, prefix="pred")


if __name__ == "__main__":
    main()
