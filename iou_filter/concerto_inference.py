#!/usr/bin/env python3
"""
Concerto 3D point cloud segmentation model wrapper.

Provides ConcertoInference class that:
- Loads Concerto backbone from HuggingFace + seg_head from checkpoint
- Prepares tile data (coordinate centering, normal estimation)
- Runs inference returning full per-point logits (not just argmax)

Reuses logic from: Concerto/demo/infer_sensaturban_scene.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# Add Concerto package to path
_concerto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Concerto')
sys.path.insert(0, _concerto_dir)

import concerto
from concerto.transform import Compose

try:
    import flash_attn  # noqa
except Exception:
    flash_attn = None


# ----------------------------
# SegHead + checkpoint utils
# (from Concerto/demo/infer_sensaturban_scene.py)
# ----------------------------

class SegHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def _extract_state_dict(ckpt):
    for k in ["state_dict", "model", "net", "module"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def _remap_keys(sd):
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


def _split_backbone_and_head(sd):
    head, backbone = {}, {}
    for k, v in sd.items():
        if k.startswith("seg_head."):
            head[k[len("seg_head."):]] = v
        else:
            backbone[k] = v
    return backbone, head


def _upcast_feat(point):
    """Reconstruct full features by upcasting through pooling hierarchy."""
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


class ConcertoInference:
    """Wrapper for Concerto 3D point cloud segmentation model."""

    def __init__(self, ckpt_path, device="cuda", grid_sample_size=0.05, normal_knn=30):
        """
        Args:
            ckpt_path: path to checkpoint with backbone weights + seg_head
            device: 'cuda' or 'cpu'
            grid_sample_size: voxel size for GridSample transform
            normal_knn: KNN neighbors for Open3D normal estimation
        """
        self.device = device
        self.normal_knn = normal_knn

        concerto.utils.set_seed(46647087)

        # Load backbone
        print("[concerto] Loading Concerto backbone...")
        if flash_attn is not None:
            self.model = concerto.load(
                "concerto_large_outdoor", repo_id="Pointcept/Concerto"
            ).to(device)
        else:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)], enable_flash=False
            )
            self.model = concerto.load(
                "concerto_large_outdoor",
                repo_id="Pointcept/Concerto",
                custom_config=custom_config,
            ).to(device)

        print(f"[concerto] Model params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        # Load seg_head from checkpoint
        print(f"[concerto] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = _remap_keys(_extract_state_dict(ckpt))
        sd_backbone, sd_head = _split_backbone_and_head(sd)

        # Load backbone weights from checkpoint
        incompat = self.model.load_state_dict(sd_backbone, strict=False)
        print(f"[concerto] backbone missing={len(incompat.missing_keys)} "
              f"unexpected={len(incompat.unexpected_keys)}")
        if incompat.missing_keys:
            print(f"  missing[:10] = {list(incompat.missing_keys)[:10]}")
        if incompat.unexpected_keys:
            print(f"  unexpected[:10] = {list(incompat.unexpected_keys)[:10]}")

        if "weight" not in sd_head:
            raise RuntimeError("No seg_head.weight found in checkpoint")

        num_classes, in_dim = sd_head["weight"].shape
        print(f"[concerto] seg_head in_dim={in_dim}, num_classes={num_classes}")

        self.seg_head = SegHead(in_dim, num_classes).to(device)
        self.seg_head.seg_head.weight.data.copy_(sd_head["weight"].to(device))
        if "bias" in sd_head:
            self.seg_head.seg_head.bias.data.copy_(sd_head["bias"].to(device))

        self.model.eval()
        self.seg_head.eval()

        # Build transform pipeline (matches val config from infer_sensaturban_scene.py)
        self.transform = Compose([
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_sample_size,
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
        ])

    def prepare_tile_data(self, tile_points, tile_size):
        """
        Prepare tile data for Concerto inference.

        Re-centers coords from BEV local [0, tile_size] to Concerto
        center-origin [-tile_size/2, tile_size/2], Z -= Z.min,
        and estimates normals via Open3D KNN.

        Args:
            tile_points: (N, 7) [x, y, z, r, g, b, class] in [0, tile_size]
            tile_size: tile size in meters

        Returns:
            dict with 'coord' (N,3), 'color' (N,3), 'normal' (N,3) numpy arrays
        """
        coord = tile_points[:, :3].copy().astype(np.float32)
        color = tile_points[:, 3:6].copy().astype(np.float32)

        # Re-center: [0, tile_size] → [-tile_size/2, tile_size/2]
        coord[:, 0] -= tile_size / 2.0
        coord[:, 1] -= tile_size / 2.0
        coord[:, 2] -= coord[:, 2].min()

        # Estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.normal_knn)
        )
        normal = np.asarray(pcd.normals, dtype=np.float32)

        return {"coord": coord, "color": color, "normal": normal}

    @torch.no_grad()
    def infer_tile_logits(self, point_data):
        """
        Run Concerto inference on prepared tile data, returning full logits.

        Args:
            point_data: dict with 'coord', 'color', 'normal' (from prepare_tile_data)

        Returns:
            (logits: (N, 13) numpy float32, pred: (N,) numpy int32)
            where N is the number of original (pre-GridSample) points
        """
        point = {
            "coord": point_data["coord"].copy(),
            "color": point_data["color"].copy(),
            "normal": point_data["normal"].copy(),
        }
        point = self.transform(point)

        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].to(self.device, non_blocking=True)

        point = self.model(point)
        point = _upcast_feat(point)

        logits_ds = self.seg_head(point.feat)       # (N_grid, 13)
        logits_full = logits_ds[point.inverse]       # (N_original, 13)

        logits_np = logits_full.cpu().numpy().astype(np.float32)
        pred_np = logits_np.argmax(axis=-1).astype(np.int32)

        return logits_np, pred_np
