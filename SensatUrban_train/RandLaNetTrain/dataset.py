import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import sys
print("Python Import Paths:")
for path in sys.path:
    print(path)
from SensatUrban.helper_ply import read_ply
    




def read_ply_data(path, with_rgb=True, with_label=True):
    data = read_ply(path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    if with_rgb and with_label:
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = data['class']
        return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
    elif with_rgb and not with_label:
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        return xyz.astype(np.float32), rgb.astype(np.uint8)
    elif not with_rgb and with_label:
        labels = data['class']
        return xyz.astype(np.float32), labels.astype(np.uint8)
    elif not with_rgb and not with_label:
        return xyz.astype(np.float32)


class SensatUrbanDataset(Dataset):
    def __init__(self, root, split,
                 voxel_size=0.2,
                 block_size=50.0,
                 max_points=200000):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.max_points = max_points

        with open(os.path.join(root, f"{split}_list.txt")) as f:
            self.files = [x.strip() for x in f]

    def voxel_downsample(self, xyz, rgb, label):
        coords = np.floor(xyz / self.voxel_size).astype(np.int32)
        _, idx = np.unique(coords, axis=0, return_index=True)
        return xyz[idx], rgb[idx], label[idx]

    def crop_block(self, xyz, rgb, label):
        # XY random 50×50m block
        minxy = xyz[:, :2].min(axis=0)
        maxxy = xyz[:, :2].max(axis=0)

        for _ in range(10):
            center = np.random.uniform(minxy, maxxy)
            x0, y0 = center - self.block_size / 2
            x1, y1 = center + self.block_size / 2

            mask = (
                (xyz[:,0] >= x0) & (xyz[:,0] <= x1) &
                (xyz[:,1] >= y0) & (xyz[:,1] <= y1)
            )
            if mask.sum() > 1000:
                xyz = xyz[mask]
                rgb = rgb[mask]
                label = label[mask]
                break

        # max points
        if len(xyz) > self.max_points:
            ids = np.random.choice(len(xyz), self.max_points, replace=False)
            xyz, rgb, label = xyz[ids], rgb[ids], label[ids]

        return xyz, rgb, label

    def __getitem__(self, idx):
         
        full_path = os.path.join(self.root, self.files[idx])
        print(full_path)
        xyz, rgb, label = read_ply_data(full_path)
        print(np.unique(label))
        print(rgb)

        xyz, rgb, label = self.voxel_downsample(xyz, rgb, label)
        xyz, rgb, label = self.crop_block(xyz, rgb, label)

        feats = np.concatenate([xyz, rgb], axis=1)

        return (
            torch.from_numpy(xyz).float(),
            torch.from_numpy(feats).float(),
            torch.from_numpy(label).long()
        )

    def __len__(self):
        return len(self.files)
