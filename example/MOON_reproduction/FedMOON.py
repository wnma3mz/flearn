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


class MOONServer(Server):
    def evaluate(self, data_lst, is_select=False):
        # 仅测试一个客户端，因为每个客户端模型一致
        if is_select == True:
            return [data_lst[0]]

        test_acc_lst = np.mean(list(map(lambda x: x["test_acc"], data_lst)), axis=0)
        test_acc = "; ".join("{:.4f}".format(x) for x in test_acc_lst)
        return test_acc


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

    def _display_iteration(self, data_loader, is_train=True):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        self.global_model.eval()
        self.global_model.to(self.device)

        loop_loss = []
        accuracy = []

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            _, pro1, output = self.model(data)
            loss1 = self.criterion(output, target)
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()

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
                loss2 = self.mu * self.criterion(logits, labels)
                loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()
            else:
                loss = loss1
            if self.display:
                data_loader.postfix = "loss: {:.4f}".format(loss.data.item())

            loop_loss.append(loss.data.item() / len(data_loader))

        # 避免占用显存
        # if is_train and self.global_model != None:
        #     self.global_model.to("cpu")
        #     for previous_net in self.previous_model_lst:
        #         previous_net.to("cpu")

        return loop_loss, accuracy


class MOONClient(Client):
    """FedMOONClient"""

    def __init__(self, conf, pre_buffer_size=1):
        # 保存以前模型的大小
        self.pre_buffer_size = pre_buffer_size
        super(MOONClient, self).__init__(conf)

    def train(self, i):
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
        self.model_trainer.global_model = copy.deepcopy(update_model)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }