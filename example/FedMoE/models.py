# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneExpertNet(nn.Module):
    """A lightweight backbone + local expert model for FedMoE."""

    def __init__(self, in_channels=1, num_classes=10, hidden_dim=128):
        super(BackboneExpertNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.backbone_proj = nn.Linear(64, hidden_dim)

        self.expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.backbone_proj(x))
        x = self.expert(x)
        return x
