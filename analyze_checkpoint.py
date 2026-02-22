#!/usr/bin/env python3
"""Analyze the ConCrETo pretrained checkpoint to understand input channel usage."""

import torch
import sys

CKPT_PATH = "/home/pavel/ITMO/NIR2/data/weights/concreto/concerto_large_outdoor.pth"

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# 1. Explore top-level keys
print("\n" + "=" * 70)
print("TOP-LEVEL KEYS IN CHECKPOINT")
print("=" * 70)
if isinstance(ckpt, dict):
    for k in sorted(ckpt.keys()):
        v = ckpt[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        elif isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {type(v).__name__}")
else:
    print(f"  Checkpoint is a {type(ckpt).__name__}, not a dict")

# 2. Find the state_dict
state_dict = None
if isinstance(ckpt, dict):
    for candidate_key in ["state_dict", "model_state_dict", "model", "net"]:
        if candidate_key in ckpt:
            state_dict = ckpt[candidate_key]
            print(f"\nUsing state_dict from key: '{candidate_key}'")
            break
    if state_dict is None:
        if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
            state_dict = ckpt
            print("\nCheckpoint itself appears to be the state_dict")
else:
    state_dict = ckpt

if state_dict is None:
    print("ERROR: Could not find state_dict in checkpoint!")
    sys.exit(1)

# 3. List all keys containing "embedding" or "stem"
print("\n" + "=" * 70)
print("KEYS RELATED TO EMBEDDING / STEM / INPUT")
print("=" * 70)
for k in sorted(state_dict.keys()):
    if any(term in k.lower() for term in ["embed", "stem", "input", "proj"]):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")

# 4. Find the target weight tensor
TARGET_CANDIDATES = [
    "embedding.stem.linear.weight",
    "backbone.embedding.stem.linear.weight",
    "module.embedding.stem.linear.weight",
    "module.backbone.embedding.stem.linear.weight",
]

weight = None
weight_key = None
for candidate in TARGET_CANDIDATES:
    if candidate in state_dict:
        weight = state_dict[candidate]
        weight_key = candidate
        break

if weight is None:
    for k, v in state_dict.items():
        if k.endswith("stem.linear.weight") and isinstance(v, torch.Tensor):
            weight = v
            weight_key = k
            break

if weight is None:
    print("\nERROR: Could not find embedding.stem.linear.weight!")
    print("\nAll keys in state_dict:")
    for k in sorted(state_dict.keys()):
        if isinstance(state_dict[k], torch.Tensor):
            print(f"  {k}: {state_dict[k].shape}")
    sys.exit(1)

print(f"\n{'=' * 70}")
print(f"FOUND: {weight_key}")
print(f"Shape: {weight.shape}")
print(f"{'=' * 70}")

out_features, in_features = weight.shape

# 5. Per-channel analysis
CHANNEL_LABELS = {
    0: "coord_x", 1: "coord_y", 2: "coord_z",
    3: "color_r", 4: "color_g", 5: "color_b",
    6: "normal_x", 7: "normal_y", 8: "normal_z",
}

print(f"\n{'=' * 70}")
print("PER INPUT-CHANNEL WEIGHT STATISTICS")
print(f"  (weight shape: [{out_features}, {in_features}] => {in_features} input channels)")
print(f"{'=' * 70}")
print(f"{'Chan':>5} {'Label':>10} {'MeanAbs':>12} {'Std':>12} {'Max':>12} {'Min':>12} {'Norm':>12}")
print("-" * 77)

for ch in range(in_features):
    col = weight[:, ch]
    mean_abs = col.abs().mean().item()
    std = col.std().item()
    max_val = col.max().item()
    min_val = col.min().item()
    norm = col.norm().item()
    label = CHANNEL_LABELS.get(ch, f"ch_{ch}")
    print(f"{ch:>5} {label:>10} {mean_abs:>12.6f} {std:>12.6f} {max_val:>12.6f} {min_val:>12.6f} {norm:>12.6f}")

# 6. Group analysis
print(f"\n{'=' * 70}")
print("GROUP ANALYSIS")
print(f"{'=' * 70}")

groups = {
    "coord (0-2)": list(range(min(3, in_features))),
    "color (3-5)": list(range(3, min(6, in_features))),
    "normal (6-8)": list(range(6, min(9, in_features))),
}
if in_features > 9:
    groups["extra (9+)"] = list(range(9, in_features))

for group_name, channels in groups.items():
    if not channels or channels[0] >= in_features:
        continue
    valid_channels = [c for c in channels if c < in_features]
    cols = weight[:, valid_channels]
    mean_abs = cols.abs().mean().item()
    std = cols.std().item()
    frobenius = cols.norm().item()
    print(f"  {group_name:>15}: mean_abs={mean_abs:.6f}, std={std:.6f}, frobenius_norm={frobenius:.6f}")

if in_features >= 6:
    coord_norm = weight[:, :3].norm().item()
    color_norm = weight[:, 3:6].norm().item()
    print(f"\n  Color/Coord norm ratio: {color_norm / coord_norm:.4f}")
    if color_norm / coord_norm < 0.01:
        print("  >>> COLOR channels have near-zero weights => model IGNORES color")
    elif color_norm / coord_norm < 0.1:
        print("  >>> COLOR channels have small weights => model uses color weakly")
    else:
        print("  >>> COLOR channels have significant weights => model USES color")

if in_features >= 9:
    coord_norm = weight[:, :3].norm().item()
    normal_norm = weight[:, 6:9].norm().item()
    print(f"  Normal/Coord norm ratio: {normal_norm / coord_norm:.4f}")
    if normal_norm / coord_norm < 0.01:
        print("  >>> NORMAL channels have near-zero weights => model IGNORES normals")
    elif normal_norm / coord_norm < 0.1:
        print("  >>> NORMAL channels have small weights => model uses normals weakly")
    else:
        print("  >>> NORMAL channels have significant weights => model USES normals")

# 7. Check for config in checkpoint
print(f"\n{'=' * 70}")
print("CONFIG / METADATA IN CHECKPOINT")
print(f"{'=' * 70}")

if isinstance(ckpt, dict):
    for k in sorted(ckpt.keys()):
        if k in ("state_dict", "model_state_dict", "model", "net", "optimizer", "optimizer_state_dict"):
            continue
        v = ckpt[k]
        if isinstance(v, dict) and len(v) < 200:
            print(f"\n  [{k}] (dict with {len(v)} keys):")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f"    {kk}: dict({len(vv)} keys)")
                elif isinstance(vv, (list, tuple)) and len(vv) > 10:
                    print(f"    {kk}: {type(vv).__name__}(len={len(vv)})")
                elif isinstance(vv, torch.Tensor):
                    print(f"    {kk}: Tensor{tuple(vv.shape)}")
                else:
                    vv_str = str(vv)
                    if len(vv_str) > 200:
                        vv_str = vv_str[:200] + "..."
                    print(f"    {kk}: {vv_str}")
        elif isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")
        elif isinstance(v, (list, tuple)):
            if len(v) <= 20:
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {type(v).__name__}(len={len(v)})")

# 8. Also check the bias
bias_key = weight_key.replace(".weight", ".bias")
if bias_key in state_dict:
    bias = state_dict[bias_key]
    print(f"\n{'=' * 70}")
    print(f"BIAS: {bias_key}, shape={bias.shape}")
    print(f"  mean={bias.mean().item():.6f}, std={bias.std().item():.6f}, "
          f"min={bias.min().item():.6f}, max={bias.max().item():.6f}")

print(f"\n{'=' * 70}")
print("DONE")
print(f"{'=' * 70}")
