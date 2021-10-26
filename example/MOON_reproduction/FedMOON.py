# coding: utf-8

import base64
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flearn.client import Client, Trainer
from flearn.server import Server


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


class AVGTrainer(Trainer):
    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class MOONTrainer(Trainer):
    """"搭配MOONClient使用"""

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(MOONTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = copy.deepcopy(self.model)
        self.previous_model_lst = []
        self.cos = nn.CosineSimilarity(dim=-1)
        # CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.5, 1, and 0.5
        self.temperature = 0.5
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 5, 1, and 1
        self.mu = 5

    def fed_loss(self, data, pro1):
        if self.global_model != None:
            # 全局与本地的对比损失，越小越好
            with torch.no_grad():
                _, pro2, _ = self.global_model(data)
            posi = self.cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # 当前轮与上一轮的对比损失，越大越好
            for previous_net in self.previous_model_lst:
                with torch.no_grad():
                    _, pro3, _ = previous_net(data)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= self.temperature
            labels = torch.zeros(data.size(0)).to(self.device).long()

            return self.mu * self.criterion(logits, labels)
        else:
            return 0

    def train(self, data_loader):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)
        return super(MOONTrainer, self).train(data_loader)

    def batch(self, data, target):
        _, pro1, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            loss += self.fed_loss(data, pro1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class MOONClient(Client):
    """MOONClient"""

    def __init__(self, conf, pre_buffer_size=1):
        super(MOONClient, self).__init__(conf)
        # 保存以前模型的大小
        self.pre_buffer_size = pre_buffer_size
        # 记录运行轮数
        self.ci = -1

    def train(self, i):
        # 每轮训练+1
        self.ci += 1
        self.train_loss, self.train_acc = self.model_trainer.loop(
            self.epoch, self.trainloader
        )
        # 权重为本地数据大小
        data_upload = self.strategy.client(
            self.model_trainer, agg_weight=len(self.trainloader)
        )
        return self._pickle_model(data_upload)

    def revice(self, i, glob_params):
        # 额外需要两类模型，glob和previous，一般情况下glob只有一个，previous也定义只有一个
        # 如果存储超过这个大小，则删除最老的模型
        while len(self.model_trainer.previous_model_lst) >= self.pre_buffer_size:
            self.model_trainer.previous_model_lst.pop(0)
        self.model_trainer.previous_model_lst.append(
            copy.deepcopy(self.model_trainer.model)
        )

        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        update_model = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model = update_model

        # 如果该客户端训练轮数不等于服务器端的训练轮数，则表示该客户端的模型本轮没有训练，则不做对比学习，并且同步进度轮数。
        if self.ci != i:
            self.model_trainer.global_model = None
            self.model_trainer.previous_model_lst = []
            self.ci = i - 1
        else:
            self.model_trainer.global_model = copy.deepcopy(update_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class ProxTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(ProxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = None
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.01, 0.001, and 0.001
        self.prox_mu = 0.01

    def fed_loss(self):
        if self.global_model != None:
            w_diff = torch.tensor(0.0, device=self.device)
            for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            return self.prox_mu / 2.0 * w_diff
        else:
            return 0

    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    def train(self, data_loader):
        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)
        return super(ProxTrainer, self).train(data_loader)


class ProxClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        update_model = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model = update_model
        self.model_trainer.global_model = copy.deepcopy(update_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class LSDTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(LSDTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.teacher_model = None
        self.mu_kd = 2
        # self.mu_kd = 0.5
        self.kd_loss = KDLoss(2)

    def train(self, data_loader):
        if self.teacher_model != None:
            self.teacher_model.eval()
            self.teacher_model.to(self.device)
        return super(LSDTrainer, self).train(data_loader)

    def batch(self, data, target):
        h, _, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            if self.teacher_model != None:
                with torch.no_grad():
                    t_h, _, t_output = self.teacher_model(data)
                h = F.normalize(h, dim=-1, p=2)
                t_h = F.normalize(h, dim=-1, p=2)
                loss2 = self.mu_kd * self.kd_loss(
                    output, t_output.detach()
                ) + self.mu_kd * self.kd_loss(h, t_h.detach())
                loss += loss2

            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class LSDClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        update_model = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model = update_model

        self.model_trainer.teacher_model = copy.deepcopy(update_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class SSDClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        bak_model = copy.deepcopy(self.model_trainer.model)
        update_model = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        # 不直接覆盖本地模型
        self.model_trainer.model = update_model

        self.model_trainer.teacher_model = copy.deepcopy(bak_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
