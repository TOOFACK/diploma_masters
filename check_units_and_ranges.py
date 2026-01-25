#!/usr/bin/env python3
import argparse
import os
import numpy as np

def load_points(input_path: str, kitti_bin: bool, npz_keys):
    ext = os.path.splitext(input_path)[1].lower()

    if kitti_bin or ext == ".bin":
        # SemanticKITTI velodyne format: float32 [x,y,z,intensity]
        arr = np.fromfile(input_path, dtype=np.float32)
        assert arr.size % 4 == 0, f"Expected multiple of 4 floats, got {arr.size}"
        arr = arr.reshape(-1, 4)
        xyz = arr[:, :3]
        feat = {"intensity": arr[:, 3]}
        return xyz, feat

    if ext == ".npz":
        data = np.load(input_path)
        coord_key, color_key, normal_key = npz_keys
        assert coord_key in data.files, f"NPZ missing coord key '{coord_key}'. Have: {data.files}"
        xyz = data[coord_key]
        feat = {}
        if color_key in data.files:
            feat["color"] = data[color_key]
        if normal_key in data.files:
            feat["normal"] = data[normal_key]
        return xyz, feat

    # open3d for ply/pcd/xyz etc.
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for non-bin/non-npz files") from e

    pcd = o3d.io.read_point_cloud(input_path)
    xyz = np.asarray(pcd.points)
    feat = {}
    if pcd.has_colors():
        feat["color"] = np.asarray(pcd.colors)
    if pcd.has_normals():
        feat["normal"] = np.asarray(pcd.normals)
    return xyz, feat


def basic_stats(xyz: np.ndarray):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    rng = maxs - mins
    center = (mins + maxs) / 2.0
    # distance to origin distribution (rough scene scale sanity)
    r = np.linalg.norm(xyz, axis=1)
    return {
        "mins": mins,
        "maxs": maxs,
        "range": rng,
        "center": center,
        "r_min": float(r.min()),
        "r_med": float(np.median(r)),
        "r_p95": float(np.percentile(r, 95)),
        "r_max": float(r.max()),
    }


def voxel_occupancy_stats(xyz: np.ndarray, grid: float):
    # voxel index
    v = np.floor(xyz / grid).astype(np.int64)
    # unique + counts
    # pack to structured for fast unique
    key = np.core.records.fromarrays(v.T, names="x,y,z", formats="i8,i8,i8")
    _, counts = np.unique(key, return_counts=True)
    return {
        "voxels": int(counts.size),
        "mean": float(counts.mean()),
        "med": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
    }


def nn_distance_stats(xyz: np.ndarray, sample_n: int = 20000, seed: int = 0):
    """
    Approximate nearest-neighbor distance using a random subset.
    Uses scipy.cKDTree if available; falls back to sklearn if available.
    """
    n = xyz.shape[0]
    if n <= 2:
        return None

    rs = np.random.RandomState(seed)
    idx = rs.choice(n, size=min(sample_n, n), replace=False)
    pts = xyz[idx]

    # Try scipy
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        d, _ = tree.query(pts, k=2, workers=-1)  # d[:,0]=0 self, d[:,1]=nn
        nn = d[:, 1]
    except Exception:
        # Try sklearn
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(xyz)
            d, _ = nbrs.kneighbors(pts)
            nn = d[:, 1]
        except Exception as e:
            raise RuntimeError("Need scipy or sklearn for NN distance stats") from e

    return {
        "sample_n": int(pts.shape[0]),
        "nn_min": float(nn.min()),
        "nn_med": float(np.median(nn)),
        "nn_p95": float(np.percentile(nn, 95)),
        "nn_p99": float(np.percentile(nn, 99)),
        "nn_max": float(nn.max()),
        "nn_mean": float(nn.mean()),
    }


def guess_units_from_nn(nn_med: float):
    """
    Very rough heuristic:
    - typical LiDAR point spacing on surfaces is centimeters (~0.02-0.20 m) depending on range.
    """
    if nn_med <= 1e-4:
        return "looks like millimeters or smaller (VERY dense / or units scaled down)"
    if nn_med <= 5e-3:
        return "looks like centimeters-ish but maybe scaled (nn_med < 0.5 cm)"
    if nn_med <= 0.3:
        return "looks like meters (reasonable LiDAR / point cloud spacing)"
    if nn_med <= 3.0:
        return "looks like meters but very sparse / or scene is huge"
    return "looks like kilometers / wrong scale likely"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .ply/.pcd/.npz or SemanticKITTI .bin")
    ap.add_argument("--kitti_bin", action="store_true", help="force SemanticKITTI bin reader")
    ap.add_argument("--npz_keys", nargs=3, default=["coord", "color", "normal"],
                    help="keys for npz: coord color normal (defaults: coord color normal)")
    ap.add_argument("--grid_list", nargs="*", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--nn_sample", type=int, default=20000)
    args = ap.parse_args()

    xyz, feat = load_points(args.input, args.kitti_bin, args.npz_keys)
    if xyz is None or len(xyz) == 0:
        raise RuntimeError("No points loaded")

    xyz = np.asarray(xyz, dtype=np.float64)
    print(f"[load] points: {xyz.shape[0]:,}  dims: {xyz.shape[1]}")
    if feat:
        print(f"[feat] keys: {list(feat.keys())}")

    st = basic_stats(xyz)
    print("\n=== Range / bounds ===")
    print(f"mins   : {st['mins']}")
    print(f"maxs   : {st['maxs']}")
    print(f"range  : {st['range']}   (dx,dy,dz)")
    print(f"center : {st['center']}")
    print("\n=== Distance to origin (rough) ===")
    print(f"r_min={st['r_min']:.4f}  r_med={st['r_med']:.4f}  r_p95={st['r_p95']:.4f}  r_max={st['r_max']:.4f}")

    print("\n=== NN distance (approx) ===")
    nn = nn_distance_stats(xyz, sample_n=args.nn_sample, seed=0)
    if nn is None:
        print("not enough points")
    else:
        print(f"sample_n={nn['sample_n']:,}")
        print(f"nn_min={nn['nn_min']:.6f}  nn_med={nn['nn_med']:.6f}  nn_p95={nn['nn_p95']:.6f}  nn_p99={nn['nn_p99']:.6f}  nn_max={nn['nn_max']:.6f}")
        print(f"nn_mean={nn['nn_mean']:.6f}")
        print(f"[units_guess] {guess_units_from_nn(nn['nn_med'])}")

    print("\n=== Voxel occupancy probe ===")
    for g in args.grid_list:
        vs = voxel_occupancy_stats(xyz, g)
        print(f"grid={g:.4f}  voxels={vs['voxels']:,}  mean={vs['mean']:.2f}  med={vs['med']:.1f}  p95={vs['p95']:.1f}  p99={vs['p99']:.1f}  max={vs['max']}")

if __name__ == "__main__":
    main()
