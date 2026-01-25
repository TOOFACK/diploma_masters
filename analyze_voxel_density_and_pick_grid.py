import argparse
import os
import numpy as np
import open3d as o3d

def load_points(path: str) -> np.ndarray:
    """
    Loads point cloud coordinates Nx3 from:
      - .ply/.pcd/.xyz via Open3D
      - KITTI .bin velodyne (float32, N x 4: x,y,z,intensity)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"]:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Unexpected points shape from {path}: {pts.shape}")
        return pts
    elif ext == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 != 0:
            raise ValueError("KITTI .bin expected float32 N*4 (x,y,z,intensity)")
        pts = arr.reshape(-1, 4)[:, :3].astype(np.float32)
        return pts
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def voxel_indices(points: np.ndarray, grid_size: float) -> np.ndarray:
    # integer voxel coords for each point
    return np.floor(points / grid_size).astype(np.int64)

def voxel_occupancy(points: np.ndarray, grid_size: float):
    """
    Returns:
      unique_vox: (M,3) voxel coords
      counts: (M,) number of points in each voxel
      inv: (N,) mapping each point -> voxel index in [0..M-1]
    """
    vox = voxel_indices(points, grid_size)
    # unique rows + inverse map
    unique_vox, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    return unique_vox, counts, inv

def summarize_counts(counts: np.ndarray) -> dict:
    counts = counts.astype(np.int64)
    q = np.quantile(counts, [0.5, 0.9, 0.95, 0.99])
    return {
        "num_voxels": int(counts.size),
        "mean": float(counts.mean()),
        "median": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": int(counts.max()),
    }

def pick_grid_size_for_target(points: np.ndarray, target_points: int, gs_min=0.01, gs_max=2.0, iters=24):
    """
    Monotonic: bigger grid_size -> fewer voxels (points after GridSample).
    Binary search to make num_voxels ~= target_points.
    """
    lo, hi = gs_min, gs_max
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        _, counts, _ = voxel_occupancy(points, mid)
        n = counts.size  # points kept after 1-per-voxel sampling
        if best is None or abs(n - target_points) < abs(best[1] - target_points):
            best = (mid, n)
        if n > target_points:
            # too many voxels => increase grid
            lo = mid
        else:
            hi = mid
    return best  # (grid_size, achieved_points)

def export_density_colored_ply(points: np.ndarray, inv: np.ndarray, counts: np.ndarray, out_path: str):
    """
    Color each point by log density of its voxel.
    """
    dens = counts[inv].astype(np.float32)
    v = np.log1p(dens)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)  # 0..1

    # simple grayscale
    colors = np.stack([v, v, v], axis=1).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)
    return out_path

def export_voxel_sample(points: np.ndarray, unique_vox: np.ndarray, inv: np.ndarray, out_path: str):
    """
    Keep ONE representative point per voxel.
    Fast method: take first occurrence per voxel using argsort on inv.
    """
    order = np.argsort(inv)
    inv_sorted = inv[order]
    # first index for each voxel
    first = np.concatenate([[0], np.where(inv_sorted[1:] != inv_sorted[:-1])[0] + 1])
    chosen_idx = order[first]
    sampled = points[chosen_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)
    return out_path, sampled.shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .ply/.pcd/.bin")
    ap.add_argument("--target_points", type=int, default=200000, help="Desired points after grid sampling (PLU)")
    ap.add_argument("--grid_list", type=float, nargs="*", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                    help="Grid sizes to probe for stats (meters)")
    ap.add_argument("--auto_grid", action="store_true", help="Auto-pick grid_size to match target_points")
    ap.add_argument("--gs_min", type=float, default=0.01)
    ap.add_argument("--gs_max", type=float, default=2.0)
    ap.add_argument("--outdir", default="./out_density")
    ap.add_argument("--export_density_ply", action="store_true", help="Save PLY colored by voxel density for chosen grid")
    ap.add_argument("--export_sample_ply", action="store_true", help="Save sampled (1-per-voxel) PLY for chosen grid")
    ap.add_argument("--visualize", action="store_true", help="Try to open Open3D viewer (needs X/forwarding)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pts = load_points(args.input)
    print(f"[load] points: {pts.shape[0]:,}")

    # probe stats for grid sizes
    print("\n=== Probe grid sizes ===")
    probe_results = []
    for gs in args.grid_list:
        _, counts, _ = voxel_occupancy(pts, gs)
        s = summarize_counts(counts)
        probe_results.append((gs, s["num_voxels"]))
        print(f"grid={gs:.4f}  voxels(kept)={s['num_voxels']:,}  mean={s['mean']:.2f}  "
              f"med={s['median']:.1f}  p95={s['p95']:.1f}  p99={s['p99']:.1f}  max={s['max']}")

    chosen_gs = None
    if args.auto_grid:
        chosen_gs, achieved = pick_grid_size_for_target(
            pts, target_points=args.target_points, gs_min=args.gs_min, gs_max=args.gs_max
        )
        print(f"\n[auto_grid] target_points={args.target_points:,} -> chosen grid_size={chosen_gs:.5f} "
              f"(kept ~{achieved:,})")
    else:
        # choose closest from probe list
        chosen_gs = min(probe_results, key=lambda x: abs(x[1] - args.target_points))[0]
        print(f"\n[choose_from_probe] target_points={args.target_points:,} -> chosen grid_size={chosen_gs:.5f}")

    unique_vox, counts, inv = voxel_occupancy(pts, chosen_gs)
    print(f"[chosen] grid={chosen_gs:.5f} -> kept(voxels)={counts.size:,}, mean pts/voxel={counts.mean():.2f}, "
          f"p99={np.quantile(counts,0.99):.1f}, max={counts.max()}")

    if args.export_density_ply:
        out_path = os.path.join(args.outdir, f"density_g{chosen_gs:.5f}.ply")
        export_density_colored_ply(pts, inv, counts, out_path)
        print(f"[write] density-colored ply: {out_path}")

    if args.export_sample_ply:
        out_path = os.path.join(args.outdir, f"sampled_g{chosen_gs:.5f}.ply")
        out_path, n = export_voxel_sample(pts, unique_vox, inv, out_path)
        print(f"[write] sampled ply (1/voxel): {out_path}  points={n:,}")

    if args.visualize:
        # visualize sampled points if requested and exists, else original colored by density if exists
        vis_path = None
        if args.export_sample_ply:
            vis_path = os.path.join(args.outdir, f"sampled_g{chosen_gs:.5f}.ply")
        elif args.export_density_ply:
            vis_path = os.path.join(args.outdir, f"density_g{chosen_gs:.5f}.ply")

        if vis_path and os.path.exists(vis_path):
            pcd = o3d.io.read_point_cloud(vis_path)
        else:
            # fallback: show original without colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
