"""
Compute normals for SensatUrban tiles in-place.
Overwrites normal.npy in each tile directory.
"""
import os
import argparse
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import open3d as o3d


def process_tile(tile_dir, knn):
    """Estimate normals for a single tile and overwrite normal.npy."""
    coord = np.load(os.path.join(tile_dir, "coord.npy")).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    np.save(os.path.join(tile_dir, "normal.npy"), normals)
    return os.path.basename(tile_dir), coord.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Tiles root (with train/val subdirs)")
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in ["train", "val"]:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] {split_dir} not found")
            continue

        tile_dirs = sorted([
            d for d in glob(os.path.join(split_dir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "coord.npy"))
        ])
        print(f"[{split}] found {len(tile_dirs)} tiles")

        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_tile, td, args.knn): td
                for td in tile_dirs
            }
            for fut in as_completed(futures):
                done += 1
                try:
                    name, npts = fut.result()
                    if done % 100 == 0 or done == len(tile_dirs):
                        print(f"  [{split}] {done}/{len(tile_dirs)} — {name} ({npts} pts)")
                except Exception as e:
                    print(f"  [ERROR] {futures[fut]}: {e}")

        print(f"[{split}] done: {done}/{len(tile_dirs)}")


if __name__ == "__main__":
    main()
