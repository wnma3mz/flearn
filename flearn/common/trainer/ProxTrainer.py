# -*- coding: utf-8 -*-
import torch

from .Trainer import Trainer


class ProxTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True, mu=1e-2) -> None:
        """
        kwargs: {
            mu : 损失的权重
        }
        """
        super(ProxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_model = None
        self.mu = mu

    def fed_loss(self):
        if self.server_model != None:
            w_diff = torch.tensor(0.0, device=self.device)
            for w, w_t in zip(self.model.parameters(), self.server_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            return self.mu / 2.0 * w_diff
        else:
            return 0
