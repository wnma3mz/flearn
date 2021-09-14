# -*- coding: utf-8 -*-
import torch
from flearn.client.train import Trainer


class ProxTrainer(Trainer):
    """"搭配ProxClient使用"""

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(ProxTrainer, self).__init__(
            model, optimizer, criterion, device, display
        )
        self.server_model = None
        self.mu = 1e-2

    def _display_iteration(self, data_loader, is_train=True):
        # assert self.optimizer.weight_decay == 0
        # assert self.optimizer.name == "sgd"
        # assert self.optimizer.momentum_factor == 0

        loop_loss = []
        accuracy = []
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            if self.display:
                data_loader.postfix = "loss: {:.4f}".format(loss.data.item())
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                # referring to https://github.com/med-air/FedBN/blob/df4a9f9c4f35696393d775f26cd4763a8c80a6f6/federated/fed_digits.py#L127
                if self.server_model != None:
                    w_diff = torch.tensor(0.0, device=self.device)
                    for w, w_t in zip(
                        self.server_model.parameters(), self.model.parameters()
                    ):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.mu / 2.0 * w_diff

                loss.backward()
                self.optimizer.step()
        return loop_loss, accuracy
