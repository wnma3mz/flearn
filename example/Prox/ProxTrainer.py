# -*- coding: utf-8 -*-
import torch
from flearn.common import Trainer


class ProxTrainer(Trainer):
    """"搭配ProxClient使用"""

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(ProxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_model = None
        self.mu = 1e-2

    def fed_loss(self):
        if self.server_model != None:
            w_diff = torch.tensor(0.0, device=self.device)
            for w, w_t in zip(self.model.parameters(), self.server_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            return self.mu / 2.0 * w_diff
        else:
            return 0
