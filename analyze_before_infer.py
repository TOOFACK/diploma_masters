# analyze_before_infer.py
import os
import json
import argparse
import numpy as np
import torch
import open3d as o3d

import concerto
from concerto.transform import Compose


BASE_TRANSFORM_CONFIG = [
    dict(type="RandomScale", scale=[0.2, 0.2]),
    dict(
        type="GridSample",
        grid_size=0.01,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    # dict(type="CenterShift", apply_z=False),
    dict(type="CenterShift", apply_z=True),

    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "color", "inverse"),
        feat_keys=("coord", "color", "normal"),
    ),
]


def load_kitti_bin(path: str):
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
    return {"coord": coord.astype(np.float32), "color": color, "normal": normal}


def load_ply(path: str):
    pcd = o3d.io.read_point_cloud(path)
    coord = np.asarray(pcd.points, dtype=np.float32)
    color = np.asarray(pcd.colors, dtype=np.float32)
    normal = np.asarray(pcd.normals, dtype=np.float32)

    if color.size == 0:
        color = np.zeros((coord.shape[0], 3), dtype=np.float32)
    if normal.size == 0:
        normal = np.zeros_like(coord, dtype=np.float32)

    if coord.size > 0:
        m = np.isfinite(coord).all(axis=1)
        coord, color, normal = coord[m], color[m], normal[m]

    return {"coord": coord, "color": color, "normal": normal}


def load_point_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_kitti_bin(path)
    if ext == ".ply":
        return load_ply(path)
    raise ValueError(f"Unsupported input extension: {ext}")


def save_ply(path: str, coord, color=None):
    coord = to_numpy(coord)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    if color is not None:
        c = to_numpy(color)

        if c.dtype != np.float32:
            c = c.astype(np.float32)

        if c.size > 0 and c.max() > 1.5:
            c = c / 255.0

        c = np.clip(c, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(c)

    o3d.io.write_point_cloud(path, pcd)



def approx_nn_stats(coord: np.ndarray, sample_n: int = 20000, seed: int = 0):
    if coord.shape[0] == 0:
        return None
    n = coord.shape[0]
    rng = np.random.default_rng(seed)
    m = min(sample_n, n)
    idx = rng.choice(n, size=m, replace=False)
    sub = coord[idx]

    # brute-ish but ok for 20k with open3d KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    kdt = o3d.geometry.KDTreeFlann(pcd)

    dists = []
    for p in sub:
        _, ii, dd = kdt.search_knn_vector_3d(p, 2)  # nearest incl self
        if len(dd) >= 2:
            dists.append(np.sqrt(dd[1]))
    if not dists:
        return None
    d = np.array(dists, dtype=np.float32)
    return {
        "nn_min": float(np.min(d)),
        "nn_med": float(np.median(d)),
        "nn_p95": float(np.percentile(d, 95)),
        "nn_p99": float(np.percentile(d, 99)),
        "nn_max": float(np.max(d)),
        "nn_mean": float(np.mean(d)),
        "sample_n": int(len(d)),
    }


def to_numpy(a):
    """Accept np.ndarray or torch.Tensor and return np.ndarray (cpu)."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a


def bounds_stats(coord):
    coord = to_numpy(coord)
    if coord is None or coord.shape[0] == 0:
        return None
    mn = coord.min(axis=0)
    mx = coord.max(axis=0)
    rg = mx - mn
    ctr = (mx + mn) / 2.0
    r = np.linalg.norm(coord, axis=1)
    return {
        "mins": mn.tolist(),
        "maxs": mx.tolist(),
        "range": rg.tolist(),
        "center": ctr.tolist(),
        "r_min": float(r.min()),
        "r_med": float(np.median(r)),
        "r_p95": float(np.percentile(r, 95)),
        "r_max": float(r.max()),
    }


def color_stats(color):
    color = to_numpy(color)
    if color is None or color.shape[0] == 0:
        return None
    c = color.astype(np.float32, copy=False)
    return {
        "min": c.min(axis=0).tolist(),
        "max": c.max(axis=0).tolist(),
        "mean": c.mean(axis=0).tolist(),
        "std": c.std(axis=0).tolist(),
        "frac_outside_0_1": float(np.mean((c < 0).any(axis=1) | (c > 1).any(axis=1))),
        "frac_zero": float(np.mean(np.all(c == 0, axis=1))),
    }


def grid_occupancy_probe(coord, grid_size: float):
    coord = to_numpy(coord)
    if coord is None or coord.shape[0] == 0:
        return None
    g = float(grid_size)
    scaled = coord / g
    grid = np.floor(scaled).astype(np.int64)

    mn = grid.min(axis=0)
    grid = grid - mn

    key = np.core.records.fromarrays(grid.T, names="x,y,z", formats="i8,i8,i8")
    _, count = np.unique(key, return_counts=True)
    count = count.astype(np.int32)

    return {
        "grid_size": g,
        "voxels": int(count.size),
        "mean": float(count.mean()),
        "med": float(np.median(count)),
        "p95": float(np.percentile(count, 95)),
        "p99": float(np.percentile(count, 99)),
        "max": int(count.max()),
    }


def validate_inverse(inverse, n_ds: int):
    inv = to_numpy(inverse)

    ok = True
    info = {
        "len": int(inv.shape[0]),
        "min": int(inv.min()) if inv.size else None,
        "max": int(inv.max()) if inv.size else None,
        "n_ds": int(n_ds),
        "frac_oob": None,
    }

    if inv.size:
        oob = (inv < 0) | (inv >= n_ds)   # numpy bool array
        info["frac_oob"] = float(oob.mean())
        ok = (info["frac_oob"] == 0.0)

    return ok, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--outdir", default="./preflight_out", type=str)
    ap.add_argument("--grid_size", default=0.06, type=float)
    ap.add_argument("--mode", default="train", choices=["train", "test"])
    ap.add_argument("--save_ply", action="store_true")
    ap.add_argument("--seed", type=int, default=46647087)
    ap.add_argument("--nn_sample", type=int, default=20000)
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--no_normal", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    concerto.utils.set_seed(args.seed)

    # ---- load
    point = load_point_file(args.input)
    if args.no_color:
        point["color"] = np.zeros_like(point["coord"], dtype=np.float32)
    if args.no_normal:
        point["normal"] = np.zeros_like(point["coord"], dtype=np.float32)

    report = {"input": args.input, "grid_size": float(args.grid_size), "mode": args.mode}

    coord0 = point["coord"]
    color0 = point["color"]

    report["pre"] = {
        "n": int(coord0.shape[0]),
        "bounds": bounds_stats(coord0),
        "nn": approx_nn_stats(coord0, sample_n=args.nn_sample, seed=args.seed),
        "color": color_stats(color0),
        "nan_inf_frac": float(np.mean(~np.isfinite(coord0).all(axis=1))) if coord0.size else 0.0,
    }
    report["pre"]["occupancy_probe"] = grid_occupancy_probe(coord0, args.grid_size)

    if args.save_ply:
        save_ply(os.path.join(args.outdir, "pre.ply"), coord0, color0)

    # ---- build transform = SAME as infer, but override grid_size and mode
    cfg = []
    for t in BASE_TRANSFORM_CONFIG:
        tt = dict(t)
        if tt.get("type") == "GridSample":
            tt["grid_size"] = float(args.grid_size)
            tt["mode"] = args.mode
        cfg.append(tt)

    transform = Compose(cfg)

    # ---- apply transform
    out = transform(point)

    # NOTE: if mode=test, GridSample returns a list of parts.
    if isinstance(out, list):
        report["post"] = {"mode_test_parts": len(out)}
        # analyze first part as representative
        out0 = out[0]
        report["post"]["part0_n_ds"] = int(out0["coord"].shape[0])
        report["post"]["part0_bounds"] = bounds_stats(out0["coord"])
        if "grid_coord" in out0:
            gc = to_numpy(out0["grid_coord"])
            report["post"]["part0_grid_coord"] = {
                "min": gc.min(axis=0).tolist(),
                "max": gc.max(axis=0).tolist(),
                "shape": list(gc.shape),
            }
        if "inverse" in out0:
            ok, inv_info = validate_inverse(out0["inverse"], int(out0["coord"].shape[0]))
            report["post"]["part0_inverse_ok"] = ok
            report["post"]["part0_inverse"] = inv_info

        if args.save_ply:
            save_ply(os.path.join(args.outdir, "post_part0_ds.ply"),
                     out0["coord"], out0.get("color", None))
    else:
        coord_ds = out["coord"]
        report["post"] = {
            "n_ds": int(coord_ds.shape[0]),
            "bounds": bounds_stats(coord_ds),
        }
        if "grid_coord" in out:
            gc = to_numpy(out["grid_coord"])
            report["post"]["grid_coord"] = {
                "min": gc.min(axis=0).tolist(),
                "max": gc.max(axis=0).tolist(),
                "shape": list(gc.shape),
            }
            # estimated physical extent implied by grid coord
            g = float(args.grid_size)
            ext = (gc.max(axis=0) - gc.min(axis=0) + 1).astype(np.float32) * g
            report["post"]["grid_extent_m"] = ext.tolist()

        if "inverse" in out:
            inv = out["inverse"]
            ok, inv_info = validate_inverse(inv, int(coord_ds.shape[0]))
            report["post"]["inverse_ok"] = ok
            report["post"]["inverse"] = inv_info

        # extra: save voxel centers PLY (helps see voxel size visually)
        if args.save_ply and "grid_coord" in out:
            gc = to_numpy(out["grid_coord"]).astype(np.float32)
            centers = (gc + 0.5) * float(args.grid_size)
            save_ply(os.path.join(args.outdir, "voxel_centers.ply"), centers, None)


        if args.save_ply:
            save_ply(os.path.join(args.outdir, "post_ds.ply"),
                     coord_ds, out.get("color", None))

    # ---- dump report
    rep_path = os.path.join(args.outdir, "report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {rep_path}")

    # ---- print key highlights for quick scanning
    pre = report["pre"]
    print("\n=== PRE ===")
    print(f"N={pre['n']}")
    b = pre["bounds"]
    if b:
        print("range(dx,dy,dz) =", np.round(np.array(b["range"]), 6).tolist())
        print("center =", np.round(np.array(b["center"]), 6).tolist())
        print("r_med/p95/max =", b["r_med"], b["r_p95"], b["r_max"])
    if pre.get("occupancy_probe"):
        o = pre["occupancy_probe"]
        print(f"occupancy@grid={o['grid_size']}: voxels={o['voxels']} mean={o['mean']:.2f} p99={o['p99']:.1f} max={o['max']}")
    if pre.get("nn"):
        nn = pre["nn"]
        print(f"nn_med={nn['nn_med']:.6f} nn_p95={nn['nn_p95']:.6f} nn_mean={nn['nn_mean']:.6f}")

    print("\n=== POST ===")
    post = report["post"]
    if "n_ds" in post:
        print(f"N_ds={post['n_ds']}")
        if post.get("grid_coord"):
            gg = post["grid_coord"]
            print("grid_coord min/max =", gg["min"], gg["max"])
            print("grid_extent_m =", np.round(np.array(post.get("grid_extent_m", [])), 6).tolist())
        if "inverse_ok" in post:
            print("inverse_ok =", post["inverse_ok"], "frac_oob =", post["inverse"].get("frac_oob"))
    else:
        print(post)

    # heuristic hints
    print("\n=== HINTS ===")
    if pre.get("occupancy_probe") and pre["occupancy_probe"]["mean"] <= 1.1:
        print("- mean pts/voxel ~1: grid_size слишком маленький или облако уже разрежено. Downsample почти не работает.")
    if pre.get("occupancy_probe") and pre["occupancy_probe"]["max"] > 200:
        print("- max pts/voxel очень большой: на плотных местах модель может страдать; увеличивай grid_size или режь тайлы.")
    if pre.get("bounds") and pre["bounds"]["r_max"] > 5000:
        print("- координаты очень большие по модулю: попробуй CenterShift/ShiftToOrigin ДО GridSample и сравни.")
    if pre.get("color") and pre["color"]["frac_outside_0_1"] > 0.01:
        print("- цвет выходит за [0,1]: NormalizeColor важен, проверь что color действительно float в 0..1 до ToTensor.")
    if ("post" in report) and isinstance(report["post"], dict) and report["post"].get("inverse_ok") is False:
        print("- inverse содержит OOB индексы: pred_ds[point.inverse] будет ломаться или давать мусор.")


if __name__ == "__main__":
    main()



# python analyze_before_infer.py \
#   --input /home/pavel/ITMO/NIR2/data/tiles/RS10/tile_00001.ply \
#   --grid_size 0.02 \
#   --mode train \
#   --outdir /home/pavel/ITMO/NIR2/data/preflight/rs10_g002 \
#   --save_ply




# python analyze_before_infer.py \
#   --input /home/pavel/ITMO/NIR2/data/SemanticKiti/dataset/dataset/sequences/15/velodyne/000015.bin \
#   --grid_size 0.02 \
#   --mode train \
#   --outdir /home/pavel/ITMO/NIR2/data/preflight/kitti_g002 \
#   --save_ply
