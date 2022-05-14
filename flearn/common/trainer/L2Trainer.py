# -*- coding: utf-8 -*-
import torch

from .Trainer import Trainer


class L2Trainer(Trainer):
    """
    对应的网络结构，需要多返回一层特征

    class LeNet5(nn.Module):
        def __init__(self, num_classes):
            super(LeNet5, self).__init__()
            self.feature_layers = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.fc1 = nn.Linear(256, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x, return_act=False):
            x = self.feature_layers(x)
            x = x.view(x.size()[0], -1)
            x = F.relu(self.fc1(x))
            act = F.relu(self.fc2(x))
            x = self.fc3(act)
            if return_act:
                return x, act
            return x
    """

    def __init__(self, model, optimizer, criterion, device, display=True, beta=1000):
        super(L2Trainer, self).__init__(model, optimizer, criterion, device, display)
        self.beta = beta  # cifar10

    def fed_loss(self):
        out = self.act

        cost = self.beta * torch.norm(out)
        return cost

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        output, act = self.model(data, return_act=True)

        self.act = act

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc
