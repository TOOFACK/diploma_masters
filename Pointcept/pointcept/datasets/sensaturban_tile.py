"""
SensatUrban tiled dataset for Pointcept.

Expected directory layout:
  data_root/
    train/
      <tile_id>/coord.npy
      <tile_id>/color.npy
      <tile_id>/normal.npy
      <tile_id>/segment.npy
    val/
      ...
"""

import os
import glob
from collections.abc import Sequence
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class SensatUrbanTileDataset(DefaultDataset):
    CLASS_NAMES = [
        "Ground",
        "Vegetation",
        "Building",
        "Wall",
        "Bridge",
        "Parking",
        "Rail",
        "Traffic Road",
        "Street Furniture",
        "Car",
        "Footpath",
        "Bike",
        "Water",
    ]

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            split_dir = os.path.join(self.data_root, split)
            candidates = sorted(glob.glob(os.path.join(split_dir, "*")))
            for candidate in candidates:
                if os.path.isdir(candidate) and os.path.exists(
                    os.path.join(candidate, "coord.npy")
                ):
                    data_list.append(candidate)
        return data_list

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        split_name = os.path.basename(os.path.dirname(data_path))
        tile_name = os.path.basename(data_path)
        return f"{split_name}/{tile_name}"
