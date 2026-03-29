# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F


class CNNBackboneExpertNet(nn.Module):
    """CNN backbone + local expert head."""

    def __init__(self, in_channels=1, num_classes=10, hidden_dim=64):
        super(CNNBackboneExpertNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.backbone_proj = nn.Linear(32 * 2 * 2, hidden_dim)
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


class MLPBackboneExpertNet(nn.Module):
    """MLP backbone + local expert head, still uses 1x8x8 input."""

    def __init__(self, in_channels=1, num_classes=10, hidden_dim=64):
        super(MLPBackboneExpertNet, self).__init__()
        input_dim = in_channels * 8 * 8
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.backbone_proj = nn.Linear(hidden_dim, hidden_dim)
        self.expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.backbone_proj(x))
        x = self.expert(x)
        return x


def build_model(model_arch, num_classes, hidden_dim):
    model_arch = model_arch.lower()
    if model_arch == "cnn":
        return CNNBackboneExpertNet(in_channels=1, num_classes=num_classes, hidden_dim=hidden_dim)
    if model_arch == "mlp":
        return MLPBackboneExpertNet(in_channels=1, num_classes=num_classes, hidden_dim=hidden_dim)
    raise ValueError("unsupported model_arch: {}".format(model_arch))
