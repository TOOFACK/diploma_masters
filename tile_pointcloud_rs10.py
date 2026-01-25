import os
import argparse
import numpy as np
import open3d as o3d


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32) if pcd.colors else None
    return pts, col


def save_tile(outdir, idx, pts, col):
    os.makedirs(outdir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if col is not None:
        pcd.colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_point_cloud(
        os.path.join(outdir, f"tile_{idx:03d}.ply"),
        pcd
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tile_xy", type=float, default=14.0)
    ap.add_argument("--tile_z", type=float, default=8.0)
    ap.add_argument("--overlap", type=float, default=2.0)
    ap.add_argument("--max_tiles", type=int, default=10)
    ap.add_argument("--z_min", type=float, default=-2.0)
    ap.add_argument("--z_max", type=float, default=6.0)
    args = ap.parse_args()

    pts, col = load_ply(args.input)

    # --- работаем в локальной системе координат ---
    xy = pts[:, :2]
    z = pts[:, 2]

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)

    step = args.tile_xy - args.overlap

    tile_id = 0
    x = min_xy[0]

    while x < max_xy[0] and tile_id < args.max_tiles:
        y = min_xy[1]
        while y < max_xy[1] and tile_id < args.max_tiles:
            x0, x1 = x, x + args.tile_xy
            y0, y1 = y, y + args.tile_xy

            mask_xy = (
                (xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            )

            # локальный Z-клип (по тротуару)
            z_loc = z[mask_xy]
            if z_loc.size == 0:
                y += step
                continue

            z0 = np.percentile(z_loc, 5) + args.z_min
            z1 = z0 + args.tile_z

            mask = mask_xy & (z >= z0) & (z <= z1)

            pts_tile = pts[mask]
            if pts_tile.shape[0] < 5000:
                y += step
                continue

            col_tile = col[mask] if col is not None else None

            save_tile(args.outdir, tile_id, pts_tile, col_tile)
            print(f"[tile {tile_id}] points={pts_tile.shape[0]} "
                  f"XY=({x0:.1f},{y0:.1f}) Z=({z0:.1f},{z1:.1f})")

            tile_id += 1
            y += step

        x += step


if __name__ == "__main__":
    main()
