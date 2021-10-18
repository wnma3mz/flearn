# coding: utf-8

import base64
import copy
import pickle
import numpy as np

import torch.nn as nn
from flearn.client import Client
from flearn.server import Server
from flearn.client.train import Trainer
import torch


class FMLServer(Server):
    def evaluate(self, data_lst, is_select=False):
        # 仅测试一个客户端，因为每个客户端模型一致
        if is_select == True:
            return [data_lst[0]]

        test_acc_lst = np.mean(list(map(lambda x: x["test_acc"], data_lst)), axis=0)
        test_acc = "; ".join("{:.4f}".format(x) for x in test_acc_lst)
        return test_acc


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
        self.local_optimizer = optim.SGD(self.local_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
        self.local_model.to(self.device)
        self.kd_loss = KDLoss(2)
        self.mu = 2

    def _key_iteration(self, data_loader, is_train=True):
        loop_loss, loop_accuracy = [], []

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            _, _, output = self.model(data)
            loss = self.criterion(output, target)

            if is_train:
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

            if self.display:
                data_loader.postfix = "loss: {:.4f}".format(loss.data.item())
            iter_loss = loss.data.item()
            iter_acc = (
                (output.data.max(1)[1] == target.data).sum().item() / len(data) * 100
            )
            loop_accuracy.append(iter_acc)
            loop_loss.append(iter_loss)

        return loop_loss, loop_accuracy


class FMLClient(Client):
    """FMLClient"""

    def __init__(self, conf, pre_buffer_size=1):
        super(FMLClient, self).__init__(conf)
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
        # decode
        data_glob_b = self.encrypt.decode(glob_params)

        # update
        update_model = self.strategy.client_revice(self.model_trainer, data_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model = update_model

        # 如果该客户端训练轮数不等于服务器端的训练轮数，则表示该客户端的模型本轮没有训练，则不做对比学习，并且同步进度轮数。
        # if self.ci != i:
        #     self.model_trainer.global_model = None
        #     self.model_trainer.previous_model_lst = []
        #     self.ci = i - 1
        # else:
        #     self.model_trainer.global_model = copy.deepcopy(update_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
