import os
import argparse
import numpy as np
import torch
import open3d as o3d

import concerto
from concerto.transform import Compose

try:
    import flash_attn
except ImportError:
    flash_attn = None


def build_transform(grid_size: float):
    # то же самое что у них, только grid_size параметризуем
    return Compose([
        dict(type="RandomScale", scale=[0.2, 0.2]),
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
    ])


def get_pca_color(feat: torch.Tensor, brightness=1.0, center=True):
    # как в оригинале
    u, s, v = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    projection = feat @ v
    projection = (
        projection[:, :3] * 0.2
        + projection[:, 3:6] * 0.2
        + projection[:, 6:9] * 0.1
        + projection[:, 9:12] * 0.5
    )
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    return color.clamp(0.0, 1.0)


def load_point_any(path: str):
    """
    Возвращает dict как ожидает concerto transform:
    coord (Nx3), color (Nx3), normal (Nx3)
    """
    if path is None:
        return concerto.data.load("sample2_outdoor")

    path = os.path.expanduser(path)
    if path.endswith(".npz"):
        data = np.load(path)
        point = {k: data[k] for k in data.files}
        # гарантируем поля
        if "color" not in point:
            point["color"] = np.zeros_like(point["coord"], dtype=np.float32)
        if "normal" not in point:
            point["normal"] = np.zeros_like(point["coord"], dtype=np.float32)
        return point

    if path.endswith(".ply") or path.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
        coord = np.asarray(pcd.points).astype(np.float32)
        if pcd.has_colors():
            color = np.asarray(pcd.colors).astype(np.float32)
            # open3d colors обычно 0..1
            if color.max() > 1.0:
                color = color / 255.0
        else:
            color = np.zeros_like(coord, dtype=np.float32)

        if pcd.has_normals():
            normal = np.asarray(pcd.normals).astype(np.float32)
        else:
            normal = np.zeros_like(coord, dtype=np.float32)

        return {"coord": coord, "color": color, "normal": normal}

    if path.endswith(".bin"):
        # SemanticKITTI velodyne: float32 (x,y,z,intensity)
        arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = arr[:, :3].astype(np.float32)
        intensity = arr[:, 3:4]
        # сделаем "color" из intensity (серый)
        color = np.repeat(intensity, 3, axis=1).astype(np.float32)
        # нормали неизвестны
        normal = np.zeros_like(coord, dtype=np.float32)
        return {"coord": coord, "color": color, "normal": normal}

    raise ValueError(f"Unsupported input format: {path}")


def load_concerto_model(mode: str, ours_ckpt: str | None, device: str):
    """
    mode:
      - baseline: загрузка с HF как у них
      - ours: загрузить базовую модель, затем частично залить веса из нашего ckpt (backbone.*)
    """
    if flash_attn is not None:
        model = concerto.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").to(device)
    else:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
        model = concerto.load("concerto_large_outdoor", repo_id="Pointcept/Concerto", custom_config=custom_config).to(device)

    if mode == "baseline":
        return model

    if mode != "ours":
        raise ValueError(f"Unknown mode: {mode}")

    if not ours_ckpt:
        raise ValueError("--ours_ckpt is required for mode=ours")

    ckpt = torch.load(ours_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # 1) берём только backbone.* (и возможно embedding.* если надо)
    # 2) убираем префиксы типа "module."
    filtered = {}
    for k, v in state_dict.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        # твой чекпойнт судя по keys: backbone.xxx и seg_head.xxx
        if kk.startswith("backbone."):
            filtered[kk] = v
        # иногда бывает что stem/embedding не под backbone, но у тебя судя по списку оно внутри backbone.embedding.*
        # поэтому этого достаточно.

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[ours] loaded backbone-only: {len(filtered)} tensors")
    print(f"[ours] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[ours] missing[:20] = {missing[:20]}")
    if len(unexpected) > 0:
        print(f"[ours] unexpected[:20] = {unexpected[:20]}")

    return model


def upcast_point_features_to_final(point):
    """
    Копия логики из их PCA-демо: поднимаем feat до "верхнего" уровня,
    чтобы фичи были сопоставимы и на исходном разрешении через inverse.
    """
    # у них: сначала 2 раза concat, потом просто поднимать
    for _ in range(2):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    return point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "ours"], default="baseline")
    parser.add_argument("--ours_ckpt", type=str, default=None, help="Path to our trained checkpoint (.pth)")
    parser.add_argument("--input", type=str, default=None, help="Path to .ply/.npz/.bin (else uses sample2_outdoor)")
    parser.add_argument("--grid_size", type=float, default=0.01)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6783)

    parser.add_argument("--wo_color", action="store_true")
    parser.add_argument("--wo_normal", action="store_true")

    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:1 / cuda / cpu")
    parser.add_argument("--save_ply", type=str, default=None, help="Output ply path (colored by PCA)")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    # device selection
    if args.device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device

    concerto.utils.set_seed(args.seed)

    model = load_concerto_model(args.mode, args.ours_ckpt, dev)
    model.eval()

    transform = build_transform(args.grid_size)
    point = load_point_any(args.input)

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"], dtype=np.float32)
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"], dtype=np.float32)

    original_coord = point["coord"].copy()
    point = transform(point)

    with torch.inference_mode():
        for key in list(point.keys()):
            if isinstance(point[key], torch.Tensor) and dev.startswith("cuda"):
                point[key] = point[key].to(dev, non_blocking=True)

        point = model(point)
        point = upcast_point_features_to_final(point)

        # PCA по point.feat (downsampled), затем раскладываем на оригинал
        pca_color = get_pca_color(point.feat, brightness=args.brightness, center=True)
        original_pca_color = pca_color[point.inverse].detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord)
    pcd.colors = o3d.utility.Vector3dVector(original_pca_color)

    if args.save_ply:
        os.makedirs(os.path.dirname(args.save_ply), exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"Saved: {args.save_ply}")

    if not args.no_show:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
