# coding: utf-8

import base64
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from flearn.client import Client
from flearn.common import Trainer


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        q = torch.softmax(target / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q) * (self.temp_factor ** 2) / input.size(0)
        # print(loss)
        return loss


import torch.optim as optim


class FMLTrainer(Trainer):
    """"搭配FMLClient使用"""

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(FMLTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.local_model = copy.deepcopy(self.model)
        self.local_optimizer = optim.SGD(
            self.local_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5
        )
        self.local_model.to(self.device)
        self.kd_loss = KDLoss(2)
        self.mu = 2

    def train(self, data_loader, epochs=1):
        self.local_model.train()
        return super(FMLTrainer, self).train(data_loader, epochs)

    # def test(self, data_loader):
    #     self.local_model.eval()
    #     return super(FMLTrainer, self).test(data_loader)

    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            _, _, local_output = self.local_model(data)
            local_loss = self.criterion(local_output, target)

            loss += self.mu * self.kd_loss(output, local_output.detach())
            local_loss += self.mu * self.kd_loss(local_output, output.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.local_optimizer.zero_grad()
            local_loss.backward()
            self.local_optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class FMLClient(Client):
    """FMLClient"""

    def __init__(self, conf):
        super(FMLClient, self).__init__(conf)
        # 记录运行轮数
        self.ci = -1

    def train(self, i):
        # 每轮训练+1
        self.ci += 1
        self.train_loss, self.train_acc = self.model_trainer.train(
            self.trainloader, self.epoch
        )
        # 权重为本地数据大小
        data_upload = self.strategy.client(
            self.model_trainer, agg_weight=len(self.trainloader)
        )
        return self._pickle_model(data_upload)

    def revice(self, i, glob_params):
        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        update_w = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model.load_state_dict(update_w)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
