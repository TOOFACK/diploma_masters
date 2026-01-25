#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def read_cloud(path: str):
    """
    Loads point cloud from .ply/.pcd using open3d.
    Returns:
      xyz: (N,3) float32
      rgb: (N,3) float32 in [0,1] or None
    """
    if o3d is None:
        raise RuntimeError("open3d is required: pip install open3d")

    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = None
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)  # already [0,1]
    return xyz, rgb


def write_npz(path: str, xyz: np.ndarray, rgb: np.ndarray | None, extra: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"coord": xyz.astype(np.float32)}
    if rgb is not None:
        payload["color"] = rgb.astype(np.float32)
    payload.update(extra)
    np.savez_compressed(path, **payload)


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray | None):
    if o3d is None:
        raise RuntimeError("open3d is required: pip install open3d")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))
    o3d.io.write_point_cloud(path, pcd)


def visualize_tiles(xyz, rgb, tiles_xyz, tiles_rgb, max_show=2):
    if o3d is None:
        print("[viz] open3d not installed, skip.")
        return
    # show global cloud (dim) + a few tiles (bright)
    p_all = o3d.geometry.PointCloud()
    p_all.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if rgb is not None:
        # dim original
        col = np.clip(rgb * 0.25, 0.0, 1.0)
    else:
        col = np.full((xyz.shape[0], 3), 0.25, dtype=np.float32)
    p_all.colors = o3d.utility.Vector3dVector(col)
    geoms = [p_all]

    for i in range(min(max_show, len(tiles_xyz))):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(tiles_xyz[i].astype(np.float32))
        if tiles_rgb[i] is not None:
            p.colors = o3d.utility.Vector3dVector(np.clip(tiles_rgb[i], 0, 1).astype(np.float32))
        else:
            p.colors = o3d.utility.Vector3dVector(np.full((tiles_xyz[i].shape[0], 3), 1.0, dtype=np.float32))
        geoms.append(p)

    o3d.visualization.draw_geometries(geoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to one SensatUrban point cloud (.ply/.pcd)")
    ap.add_argument("--outdir", required=True, help="Directory to write tiles")
    ap.add_argument("--tile", type=float, default=50.0, help="Tile size in meters (square in XY)")
    ap.add_argument("--stride", type=float, default=None, help="Stride in meters (default = tile, no overlap)")
    ap.add_argument("--min_points", type=int, default=20000, help="Skip tiles with fewer points")
    ap.add_argument("--z_min", type=float, default=None, help="Optional Z filter min")
    ap.add_argument("--z_max", type=float, default=None, help="Optional Z filter max")
    ap.add_argument("--save_ply", action="store_true", help="Also write each tile as .ply")
    ap.add_argument("--visualize", action="store_true", help="Visualize a couple of tiles on top of the scene")
    ap.add_argument("--max_tiles", type=int, default=0, help="If >0, stop after writing this many tiles (debug)")
    args = ap.parse_args()

    tile = float(args.tile)
    stride = float(args.stride) if args.stride is not None else tile
    assert stride > 0 and tile > 0

    xyz, rgb = read_cloud(args.input)

    # optional Z filter
    m = np.ones((xyz.shape[0],), dtype=bool)
    if args.z_min is not None:
        m &= (xyz[:, 2] >= float(args.z_min))
    if args.z_max is not None:
        m &= (xyz[:, 2] <= float(args.z_max))
    xyz = xyz[m]
    rgb = rgb[m] if rgb is not None else None

    if xyz.shape[0] == 0:
        raise RuntimeError("No points after filtering")

    # bounds in XY
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    nx = int(math.ceil((x_max - x_min - tile) / stride)) + 1
    ny = int(math.ceil((y_max - y_min - tile) / stride)) + 1
    nx = max(nx, 1)
    ny = max(ny, 1)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_npz_dir = os.path.join(args.outdir, base, "npz")
    out_ply_dir = os.path.join(args.outdir, base, "ply")

    written = 0
    kept_xyz = []
    kept_rgb = []

    for iy in range(ny):
        y0 = y_min + iy * stride
        y1 = y0 + tile
        for ix in range(nx):
            x0 = x_min + ix * stride
            x1 = x0 + tile

            mask = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            n = int(mask.sum())
            if n < args.min_points:
                continue

            t_xyz = xyz[mask]
            t_rgb = rgb[mask] if rgb is not None else None

            # save tile metadata for downstream stitching
            meta = {
                "tile_x0": np.float32(x0),
                "tile_x1": np.float32(x1),
                "tile_y0": np.float32(y0),
                "tile_y1": np.float32(y1),
                "tile_ix": np.int32(ix),
                "tile_iy": np.int32(iy),
            }

            name = f"{base}_ix{ix:03d}_iy{iy:03d}_n{n}"
            npz_path = os.path.join(out_npz_dir, name + ".npz")
            write_npz(npz_path, t_xyz, t_rgb, meta)

            if args.save_ply:
                ply_path = os.path.join(out_ply_dir, name + ".ply")
                write_ply(ply_path, t_xyz, t_rgb)

            written += 1
            if args.visualize and len(kept_xyz) < 2:
                kept_xyz.append(t_xyz)
                kept_rgb.append(t_rgb)

            if args.max_tiles > 0 and written >= args.max_tiles:
                break
        if args.max_tiles > 0 and written >= args.max_tiles:
            break

    print(f"[ok] written tiles: {written}")
    print(f"      out npz: {out_npz_dir}")
    if args.save_ply:
        print(f"      out ply: {out_ply_dir}")

    if args.visualize and written > 0 and len(kept_xyz) > 0:
        visualize_tiles(xyz, rgb, kept_xyz, kept_rgb, max_show=len(kept_xyz))


if __name__ == "__main__":
    main()
