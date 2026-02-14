import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(SCRIPT_DIR, "preprocess")
sys.path.insert(0, PREPROCESS_DIR)

from point_EDA_31 import SensatUrbanEDA
import helper_image as hi


LABEL_COLOR_MAP14 = [
    [0, 0, 0], [255, 248, 220], [220, 220, 220], [139, 71, 38],
    [238, 197, 145], [70, 130, 180], [179, 238, 58], [110, 139, 61],
    [105, 105, 105], [0, 0, 128], [205, 92, 92], [244, 164, 96],
    [147, 112, 219], [255, 228, 225],
]

MEAN_4 = (0.485, 0.456, 0.406, 0.115)
STD_4 = (0.229, 0.224, 0.225, 0.067)
MEAN_3 = MEAN_4[:3]
STD_3 = STD_4[:3]


def _ensure_out_dirs(out_dir: str) -> None:
    for sub in ("rgb", "alt", "rgbd", "pred_cls", "pred_vis"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def _build_rgbd_tile(pts, grid_scale, grid_step, label_color_map, do_complete=True):
    alt, rgb, _ = hi.project_3d_bev_img3(pts.T, grid_scale, grid_step, label_color_map)

    if do_complete:
        n_loop = 3
        alt = hi.complete2d(alt, n_loop)
        for c in range(3):
            rgb[:, :, c] = hi.complete2d(rgb[:, :, c], n_loop)

    # sanitize and scale if colors are in 0..1
    rgb = np.nan_to_num(rgb, nan=0.0)
    alt = np.nan_to_num(alt, nan=0.0)
    rgb[rgb < 0] = 0
    alt[alt < 0] = 0

    if rgb.size and rgb.max() <= 1.0:
        rgb = rgb * 255.0
    if alt.size and alt.max() <= 1.0:
        alt = alt * 255.0

    alt = np.clip(alt, 0, 255).astype(np.uint8)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgbd = np.dstack([rgb, alt])
    return rgbd, rgb, alt


def _normalize_to_tensor(img, in_channels):
    img = img.astype(np.float32) / 255.0
    if in_channels == 4:
        img -= np.array(MEAN_4, dtype=np.float32)
        img /= np.array(STD_4, dtype=np.float32)
    else:
        img -= np.array(MEAN_3, dtype=np.float32)
        img /= np.array(STD_3, dtype=np.float32)
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def _colorize_pred(pred):
    h, w = pred.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(LABEL_COLOR_MAP14):
        vis[pred == cls] = color[::-1]  # RGB -> BGR for cv2
    return vis


def _remap_axes_xyz(xyz, coord_order, bev_axes, height_axis):
    axes = {"x": 0, "y": 1, "z": 2}
    if sorted(coord_order) != ["x", "y", "z"]:
        raise ValueError("coord_order must be a permutation of xyz")
    # reorder input columns into canonical x,y,z
    reorder = [coord_order.index("x"), coord_order.index("y"), coord_order.index("z")]
    pts_xyz = xyz[:, reorder].copy()
    bx = axes[bev_axes[0]]
    by = axes[bev_axes[1]]
    hz = axes[height_axis]
    if len({bx, by, hz}) != 3:
        raise ValueError("bev_axes and height_axis must be distinct")
    # project into expected order: [x, y, z]
    pts_xyz[:, [0, 1, 2]] = pts_xyz[:, [bx, by, hz]]
    return pts_xyz


def _adapt_segformer_input_channels(model, in_channels):
    if model.config.num_channels == in_channels:
        return
    model.config.num_channels = in_channels
    proj = model.segformer.encoder.patch_embeddings[0].proj
    if proj.in_channels == in_channels:
        return

    new_proj = torch.nn.Conv2d(
        in_channels,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    with torch.no_grad():
        new_proj.weight.zero_()
        if in_channels >= proj.in_channels:
            new_proj.weight[:, :proj.in_channels].copy_(proj.weight)
            mean_w = proj.weight.mean(dim=1, keepdim=True)
            for c in range(proj.in_channels, in_channels):
                new_proj.weight[:, c:c + 1].copy_(mean_w)
        else:
            new_proj.weight.copy_(proj.weight[:, :in_channels])
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)

    model.segformer.encoder.patch_embeddings[0].proj = new_proj


def _load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model_state = model.state_dict()
    new_state = {}
    model_has_module = next(iter(model_state)).startswith("module.")
    for k, v in state.items():
        kk = k
        if kk.startswith("module.") and not model_has_module:
            kk = kk[len("module."):]
        elif (not kk.startswith("module.")) and model_has_module:
            kk = "module." + kk
        new_state[kk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[ckpt] missing keys:", len(missing))
    if unexpected:
        print("[ckpt] unexpected keys:", len(unexpected))


def run_inference(args):
    _ensure_out_dirs(args.out_dir)

    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=14,
        ignore_mismatched_sizes=True,
    )
    _adapt_segformer_input_channels(model, args.in_channels)

    device = "cuda" if args.cuda else "cpu"
    if args.cuda:
        model = model.cuda()

    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    _load_checkpoint(model, args.resume, device)
    if args.cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()

    sensat = SensatUrbanEDA()
    sensat.grids_scale = args.grid_scale
    sensat.grids_size = args.grid_size
    sensat.grids_step = args.grid_step

    ply_data = sensat.load_points(args.ply, reformat=True)
    ply_data = ply_data.copy()
    ply_data[:, :3] = _remap_axes_xyz(ply_data[:, :3], args.coord_order, args.bev_axes, args.height_axis)
    if args.debug and ply_data.shape[1] >= 6:
        rgb_raw = ply_data[:, 3:6]
        print("[debug] raw rgb min/max:", rgb_raw.min(axis=0), rgb_raw.max(axis=0))
    grids = sensat.grid_generator(ply_data, sensat.grids_size, sensat.grids_step, False)
    ply_id = os.path.splitext(os.path.basename(args.ply))[0]

    for x_idx, y_idx, pts in tqdm(grids, desc="tiles"):
        pts_use = pts
        z_min = float(pts_use[:, 2].min())
        z_max = float(pts_use[:, 2].max())
        if z_min < -999.0:
            pts_use = pts_use.copy()
            pts_use[:, 2] = pts_use[:, 2] - z_min
        if args.swap_xy:
            if pts_use is pts:
                pts_use = pts.copy()
            pts_use[:, [0, 1]] = pts_use[:, [1, 0]]
        if args.debug:
            print("[debug] tile", x_idx, y_idx, "z min/max", z_min, z_max)

        rgbd, rgb, alt = _build_rgbd_tile(
            pts_use,
            sensat.grids_scale,
            sensat.grids_step,
            sensat.label_color_map,
            do_complete=not args.no_complete,
        )
        if args.transpose_bev:
            rgb = rgb.transpose(1, 0, 2)
            alt = alt.T
            rgbd = np.dstack([rgb, alt])
        if args.debug:
            nonzero = np.count_nonzero(rgb)
            print("[debug] tile", x_idx, y_idx, "rgb min/max", rgb.min(), rgb.max(), "nonzero", nonzero)

        if args.crop_size and rgbd.shape[0] != args.crop_size:
            rgbd = cv2.resize(rgbd, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.resize(rgb, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
            alt = cv2.resize(alt, (args.crop_size, args.crop_size), interpolation=cv2.INTER_NEAREST)

        if args.in_channels == 3:
            inp = rgb
        else:
            inp = rgbd
        tensor = _normalize_to_tensor(inp, args.in_channels)
        if args.cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            outputs = model(tensor)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if logits.shape[-2:] != tensor.shape[-2:]:
                logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        vis = _colorize_pred(pred)

        tile_id = "{}_{}_{}".format(ply_id, x_idx, y_idx)
        cv2.imwrite(os.path.join(args.out_dir, "rgb", "{}.png".format(tile_id)), rgb)
        cv2.imwrite(os.path.join(args.out_dir, "alt", "{}.png".format(tile_id)), alt)
        cv2.imwrite(os.path.join(args.out_dir, "rgbd", "{}.png".format(tile_id)), rgbd)
        cv2.imwrite(os.path.join(args.out_dir, "pred_cls", "{}.png".format(tile_id)), pred)
        cv2.imwrite(os.path.join(args.out_dir, "pred_vis", "{}.png".format(tile_id)), vis)


def main():
    parser = argparse.ArgumentParser(description="BEV inference for a single SensatUrban PLY")
    parser.add_argument("--ply", type=str, required=True, help="path to a single .ply file")
    parser.add_argument("--resume", type=str, required=True, help="checkpoint path")
    parser.add_argument("--out-dir", type=str, default="outputs/ply_infer", help="output directory")
    parser.add_argument("--grid-scale", type=float, default=0.05, help="grid scale in meters")
    parser.add_argument("--grid-size", type=int, default=25, help="grid size in meters")
    parser.add_argument("--grid-step", type=int, default=25, help="grid step in meters")
    parser.add_argument("--crop-size", type=int, default=500, help="tile size for inference")
    parser.add_argument("--in-channels", type=int, default=3, choices=[3, 4], help="model input channels")
    parser.add_argument("--swap-xy", action="store_true", default=False, help="swap X/Y axes before BEV")
    parser.add_argument("--transpose-bev", action="store_true", default=False, help="transpose BEV image axes")
    parser.add_argument("--coord-order", type=str, default="yzx", help="input coord order, e.g. xyz, xzy, yxz")
    parser.add_argument("--bev-axes", type=str, default="xy", help="axes for BEV plane, e.g. xy, xz, yz")
    parser.add_argument("--height-axis", type=str, default="z", help="axis used as height, one of x,y,z")
    parser.add_argument("--no-complete", action="store_true", default=False, help="disable completion for BEV")
    parser.add_argument("--debug", action="store_true", default=False, help="print debug stats")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma-separated GPU ids")

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
    else:
        args.gpu_ids = []

    run_inference(args)


if __name__ == "__main__":
    main()
