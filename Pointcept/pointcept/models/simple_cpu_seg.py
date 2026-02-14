import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from .builder import MODELS


@MODELS.register_module()
class SimpleCPUSegmentor(nn.Module):
    """
    Минимальная CPU-friendly модель для smoke-теста датасета/тренировочного цикла.
    Ожидает входной ключ `feat` формы [N, C].
    """

    def __init__(self, num_classes=13, in_channels=9, hidden_channels=64, criteria=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes),
        )
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        feat = input_dict["feat"].float()
        seg_logits = self.net(feat)
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        return dict(seg_logits=seg_logits)
