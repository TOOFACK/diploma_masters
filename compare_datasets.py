#!/usr/bin/env python3
"""Comprehensive comparison of SensatUrban tiles vs SemanticKITTI data."""

import numpy as np
import os

# ============================================================
# Helper functions
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    print(f"\n--- {title} ---")


def describe_array(name, arr):
    """Print detailed statistics for a numpy array."""
    print(f"\n  [{name}]")
    print(f"    shape: {arr.shape}")
    print(f"    dtype: {arr.dtype}")
    print(f"    total elements: {arr.size}")

    if arr.ndim == 2:
        axis_labels = ['x', 'y', 'z'] if arr.shape[1] <= 3 else [f'dim{i}' for i in range(arr.shape[1])]
        if arr.shape[1] == 3 and name == 'color':
            axis_labels = ['R', 'G', 'B']
        for i in range(arr.shape[1]):
            col = arr[:, i]
            mn, mx = col.min(), col.max()
            print(f"    {axis_labels[i]:>5s}: min={mn:12.4f}  max={mx:12.4f}  range={mx - mn:12.4f}  mean={col.mean():12.4f}  std={col.std():12.4f}")
    elif arr.ndim == 1:
        mn, mx = arr.min(), arr.max()
        print(f"    min={mn}  max={mx}  range={mx - mn}")
        if np.issubdtype(arr.dtype, np.floating):
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")

    # Overall stats
    print(f"    overall min={arr.min():.6f}  max={arr.max():.6f}")
    print(f"    overall mean={arr.mean():.6f}  std={arr.std():.6f}")


def compute_density(coord):
    """Compute points per unit volume."""
    mins = coord.min(axis=0)
    maxs = coord.max(axis=0)
    ranges = maxs - mins
    volume = np.prod(ranges)
    n_points = coord.shape[0]
    density = n_points / volume if volume > 0 else float('inf')
    return density, volume, ranges


# ============================================================
# 1. Load SensatUrban tile
# ============================================================

print_header("SENSATURBAN TILE")
su_dir = "/home/pavel/ITMO/NIR2/Pointcept/data/sensaturban_tiles/train/birmingham_block_0_tile_256_129/"

su_coord = np.load(os.path.join(su_dir, "coord.npy"))
su_color = np.load(os.path.join(su_dir, "color.npy"))
su_normal = np.load(os.path.join(su_dir, "normal.npy"))
su_segment = np.load(os.path.join(su_dir, "segment.npy"))

print(f"\nTile: birmingham_block_0_tile_256_129")
print(f"Total number of points: {su_coord.shape[0]:,}")

describe_array("coord", su_coord)
describe_array("color", su_color)
describe_array("normal", su_normal)

# Segment info
print(f"\n  [segment]")
print(f"    shape: {su_segment.shape}")
print(f"    dtype: {su_segment.dtype}")
unique_vals, counts = np.unique(su_segment, return_counts=True)
print(f"    unique values ({len(unique_vals)}): {unique_vals.tolist()}")
print(f"    counts:                     {counts.tolist()}")
print(f"    percentages:")
for v, c in zip(unique_vals, counts):
    print(f"      class {v:3d}: {c:8,} points ({100.0 * c / su_segment.shape[0]:6.2f}%)")

# Normal: % nonzero
nonzero_mask = np.any(su_normal != 0, axis=1)
print(f"\n  [normal] % nonzero rows: {100.0 * nonzero_mask.sum() / su_normal.shape[0]:.2f}%")
print(f"  [normal] % zero rows:    {100.0 * (~nonzero_mask).sum() / su_normal.shape[0]:.2f}%")

# Point density
density, volume, ranges = compute_density(su_coord)
print(f"\n  [density]")
print(f"    Bounding box ranges (x, y, z): {ranges}")
print(f"    Bounding box volume: {volume:,.2f} cubic units")
print(f"    Points per unit volume: {density:,.4f}")

# ============================================================
# 2. SensatUrban color analysis
# ============================================================

print_subheader("SensatUrban Color Analysis")

# Are color values integers stored as float?
if np.issubdtype(su_color.dtype, np.floating):
    is_whole = np.all(su_color == np.floor(su_color))
    print(f"  Color dtype is float: {su_color.dtype}")
    print(f"  All values are whole numbers (ints stored as float)? {is_whole}")
    if not is_whole:
        # Check what fraction are whole
        frac_whole = np.mean(su_color == np.floor(su_color))
        print(f"  Fraction of values that are whole numbers: {frac_whole:.6f}")
else:
    print(f"  Color dtype is integer: {su_color.dtype}")

# Color range distribution
flat_color = su_color.flatten()
in_0_1 = np.sum((flat_color >= 0) & (flat_color <= 1))
in_1_128 = np.sum((flat_color > 1) & (flat_color <= 128))
in_128_255 = np.sum((flat_color > 128) & (flat_color <= 255))
above_255 = np.sum(flat_color > 255)
below_0 = np.sum(flat_color < 0)

print(f"\n  Color range distribution (all channels flattened):")
print(f"    [< 0]:       {below_0:10,}  ({100.0 * below_0 / flat_color.size:.4f}%)")
print(f"    [0, 1]:      {in_0_1:10,}  ({100.0 * in_0_1 / flat_color.size:.4f}%)")
print(f"    (1, 128]:    {in_1_128:10,}  ({100.0 * in_1_128 / flat_color.size:.4f}%)")
print(f"    (128, 255]:  {in_128_255:10,}  ({100.0 * in_128_255 / flat_color.size:.4f}%)")
print(f"    (> 255):     {above_255:10,}  ({100.0 * above_255 / flat_color.size:.4f}%)")

# Also show a finer histogram
print(f"\n  Color value histogram (10 bins across actual range):")
hist, bin_edges = np.histogram(flat_color, bins=10)
for i in range(len(hist)):
    print(f"    [{bin_edges[i]:8.2f}, {bin_edges[i+1]:8.2f}): {hist[i]:10,}")

# Per-channel stats
print(f"\n  Per-channel unique value counts:")
for ch, ch_name in enumerate(['R', 'G', 'B']):
    n_unique = len(np.unique(su_color[:, ch]))
    print(f"    {ch_name}: {n_unique} unique values")


# ============================================================
# 3. Load SemanticKITTI scan
# ============================================================

print_header("SEMANTICKITTI SCAN")

kitti_vel_dir = "/home/pavel/ITMO/NIR2/data/SemanticKiti/dataset/dataset/sequences/00/velodyne/"
kitti_label_dir = "/home/pavel/ITMO/NIR2/data/SemanticKiti/dataset/dataset/sequences/00/labels/"

kitti_bin_file = os.path.join(kitti_vel_dir, "000000.bin")
kitti_label_file = os.path.join(kitti_label_dir, "000000.label")

# Load point cloud: float32 [x, y, z, intensity]
kitti_raw = np.fromfile(kitti_bin_file, dtype=np.float32).reshape(-1, 4)
kitti_coord = kitti_raw[:, :3]
kitti_intensity = kitti_raw[:, 3]

print(f"\nScan: 000000.bin")
print(f"Total number of points: {kitti_coord.shape[0]:,}")

describe_array("coord", kitti_coord)

print(f"\n  [intensity]")
print(f"    shape: {kitti_intensity.shape}")
print(f"    dtype: {kitti_intensity.dtype}")
print(f"    min={kitti_intensity.min():.6f}  max={kitti_intensity.max():.6f}")
print(f"    mean={kitti_intensity.mean():.6f}  std={kitti_intensity.std():.6f}")

# Color: KITTI has no color, dataset class returns zeros
kitti_color = np.zeros((kitti_coord.shape[0], 3), dtype=np.float32)
describe_array("color (synthetic zeros)", kitti_color)

# Normal: KITTI has no normals, dataset class returns zeros
kitti_normal = np.zeros((kitti_coord.shape[0], 3), dtype=np.float32)
print(f"\n  [normal (synthetic zeros)]")
print(f"    shape: {kitti_normal.shape}")
print(f"    dtype: {kitti_normal.dtype}")
print(f"    All zeros (no normals in LiDAR data)")
nonzero_mask_k = np.any(kitti_normal != 0, axis=1)
print(f"    % nonzero rows: {100.0 * nonzero_mask_k.sum() / kitti_normal.shape[0]:.2f}%")

# Labels
kitti_segment = None
if os.path.exists(kitti_label_file):
    kitti_labels_raw = np.fromfile(kitti_label_file, dtype=np.uint32)
    # Lower 16 bits = semantic label, upper 16 bits = instance id
    kitti_segment = (kitti_labels_raw & 0xFFFF).astype(np.int64)
    print(f"\n  [segment (semantic label, lower 16 bits)]")
    print(f"    shape: {kitti_segment.shape}")
    print(f"    dtype: {kitti_segment.dtype}")
    unique_vals_k, counts_k = np.unique(kitti_segment, return_counts=True)
    print(f"    unique values ({len(unique_vals_k)}): {unique_vals_k.tolist()}")
    print(f"    counts:                     {counts_k.tolist()}")
    print(f"    percentages:")
    for v, c in zip(unique_vals_k, counts_k):
        print(f"      class {v:3d}: {c:8,} points ({100.0 * c / kitti_segment.shape[0]:6.2f}%)")
else:
    print(f"\n  [segment] Label file not found: {kitti_label_file}")

# Point density
density_k, volume_k, ranges_k = compute_density(kitti_coord)
print(f"\n  [density]")
print(f"    Bounding box ranges (x, y, z): {ranges_k}")
print(f"    Bounding box volume: {volume_k:,.2f} cubic units")
print(f"    Points per unit volume: {density_k:,.4f}")


# ============================================================
# 4. Comparison
# ============================================================

print_header("COMPARISON: SensatUrban vs SemanticKITTI")

print(f"\n  {'Metric':<40s} {'SensatUrban':>20s} {'SemanticKITTI':>20s}")
print(f"  {'-'*40} {'-'*20} {'-'*20}")

print(f"  {'Number of points':<40s} {su_coord.shape[0]:>20,} {kitti_coord.shape[0]:>20,}")

# Coordinate ranges
su_ranges = su_coord.max(axis=0) - su_coord.min(axis=0)
k_ranges = kitti_coord.max(axis=0) - kitti_coord.min(axis=0)

print(f"  {'Coord X range':<40s} {su_ranges[0]:>20.2f} {k_ranges[0]:>20.2f}")
print(f"  {'Coord Y range':<40s} {su_ranges[1]:>20.2f} {k_ranges[1]:>20.2f}")
print(f"  {'Coord Z range':<40s} {su_ranges[2]:>20.2f} {k_ranges[2]:>20.2f}")

print(f"  {'Coord X min':<40s} {su_coord[:,0].min():>20.2f} {kitti_coord[:,0].min():>20.2f}")
print(f"  {'Coord X max':<40s} {su_coord[:,0].max():>20.2f} {kitti_coord[:,0].max():>20.2f}")
print(f"  {'Coord Y min':<40s} {su_coord[:,1].min():>20.2f} {kitti_coord[:,1].min():>20.2f}")
print(f"  {'Coord Y max':<40s} {su_coord[:,1].max():>20.2f} {kitti_coord[:,1].max():>20.2f}")
print(f"  {'Coord Z min':<40s} {su_coord[:,2].min():>20.2f} {kitti_coord[:,2].min():>20.2f}")
print(f"  {'Coord Z max':<40s} {su_coord[:,2].max():>20.2f} {kitti_coord[:,2].max():>20.2f}")

print(f"  {'Bounding box volume':<40s} {volume:>20,.2f} {volume_k:>20,.2f}")
print(f"  {'Point density (pts/unit^3)':<40s} {density:>20.4f} {density_k:>20.4f}")

print(f"  {'Coord dtype':<40s} {str(su_coord.dtype):>20s} {str(kitti_coord.dtype):>20s}")
print(f"  {'Color dtype':<40s} {str(su_color.dtype):>20s} {'N/A (zeros)':>20s}")
print(f"  {'Normal dtype':<40s} {str(su_normal.dtype):>20s} {'N/A (zeros)':>20s}")
print(f"  {'Segment dtype':<40s} {str(su_segment.dtype):>20s} {str(kitti_segment.dtype) if kitti_segment is not None else 'N/A':>20s}")

print(f"  {'Has real color':<40s} {'YES':>20s} {'NO':>20s}")
print(f"  {'Has real normals':<40s} {'YES':>20s} {'NO':>20s}")
print(f"  {'Has intensity':<40s} {'NO':>20s} {'YES':>20s}")

print(f"  {'Number of semantic classes':<40s} {len(unique_vals):>20d} {len(unique_vals_k) if kitti_segment is not None else 0:>20d}")

# Coordinate scale comparison
print_subheader("Coordinate Scale Comparison")
print(f"\n  SensatUrban coordinates appear to be in METERS (aerial/terrestrial laser scan)")
print(f"    Typical X range: {su_ranges[0]:.2f} m")
print(f"    Typical Y range: {su_ranges[1]:.2f} m")
print(f"    Typical Z range: {su_ranges[2]:.2f} m")
print(f"    This is a tile of ~{su_ranges[0]:.0f}m x {su_ranges[1]:.0f}m footprint")

print(f"\n  SemanticKITTI coordinates are in METERS (vehicle-mounted LiDAR)")
print(f"    Typical X range: {k_ranges[0]:.2f} m")
print(f"    Typical Y range: {k_ranges[1]:.2f} m")
print(f"    Typical Z range: {k_ranges[2]:.2f} m")
print(f"    This is a ~{k_ranges[0]:.0f}m x {k_ranges[1]:.0f}m LiDAR sweep")

print(f"\n  Scale ratio (SensatUrban / SemanticKITTI):")
print(f"    X: {su_ranges[0] / k_ranges[0]:.2f}x")
print(f"    Y: {su_ranges[1] / k_ranges[1]:.2f}x")
print(f"    Z: {su_ranges[2] / k_ranges[2]:.2f}x")
print(f"    Volume: {volume / volume_k:.2f}x")
print(f"    Points: {su_coord.shape[0] / kitti_coord.shape[0]:.2f}x")

# Key differences summary
print_subheader("Key Differences Summary")
print("""
  1. ACQUISITION: SensatUrban = aerial/terrestrial survey (dense, uniform coverage)
                  SemanticKITTI = vehicle-mounted rotating LiDAR (sparse, radial pattern)

  2. DENSITY:     SensatUrban is much denser per unit area (survey-grade)
                  SemanticKITTI density decreases with distance from sensor

  3. FEATURES:    SensatUrban has RGB color + surface normals
                  SemanticKITTI has intensity only (no color, no normals)

  4. SCALE:       Tiles vs single 360-degree sweeps
                  Very different spatial extents and point counts

  5. CLASSES:     Different semantic label schemes and class counts
""")

print("Done.")
