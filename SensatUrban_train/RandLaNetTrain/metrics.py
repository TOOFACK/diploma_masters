import numpy as np
import torch

def compute_iou(pred, label, num_classes=13):
    ious = []
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    for c in range(num_classes):
        p = pred == c
        l = label == c

        inter = (p & l).sum()
        union = (p | l).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)

    return ious

def compute_miou(ious):
    return np.nanmean(ious)
