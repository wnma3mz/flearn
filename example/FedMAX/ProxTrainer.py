# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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


class MaxTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(MaxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.beta = 1000  # cifar10
        self.kldiv = nn.KLDivLoss(reduce=True)

    def fed_loss(self):
        out = self.act

        zero_mat = torch.zeros(out.size()).cuda()
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)

        cost = self.beta * self.kldiv(logsoftmax(out), softmax(zero_mat))

        return cost

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        output, act = self.model(data, return_act=True)

        self.act = act

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class L2Trainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(L2Trainer, self).__init__(model, optimizer, criterion, device, display)
        self.beta = 1000  # cifar10

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
