# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
import copy


class Trainer(ABC):
    # this flag allows you to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, criterion, device, display=True):
        """模型训练器

        Args:
            model :       torchvision.models
                          模型

            optimizer :   torch.optim
                          优化器

            criterion :   torch.nn.modules.loss
                          损失函数

            device :      torch.device
                          指定gpu还是cpu

            display :     bool (default: `True`)
                          是否显示过程
        """
        self.model = model
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.optimizer = optimizer
        self.criterion = criterion
        self.display = display
        self.model.to(self.device)
        self.model_o = copy.deepcopy(self.weight)

    def _display_iteration(self, data_loader, is_train=True):
        loop_loss, accuracy = [], []
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
                loss.backward()
                self.optimizer.step()
        return loop_loss, accuracy

    def _iteration(self, data_loader, is_train=True):
        if self.display:
            with tqdm(data_loader, ncols=80, postfix="loss: *.****") as t:
                loop_loss, accuracy = self._display_iteration(t, is_train)
        else:
            loop_loss, accuracy = self._display_iteration(data_loader, is_train)

        return np.sum(loop_loss), np.sum(accuracy) / len(data_loader.dataset) * 100

    def train(self, data_loader):
        self.model_o = copy.deepcopy(self.weight)
        self.model.train()
        with torch.enable_grad():
            loop_loss, accuracy = self._iteration(data_loader)
        return loop_loss, accuracy

    def test(self, data_loader):
        """
        Args:
            data_loader :  torch.utils.data
                           测试集

        Returns:
            float : loop_loss
                    损失值

            float : accuracy
                    准确率
        """
        self.model.eval()
        with torch.no_grad():
            loop_loss, accuracy = self._iteration(data_loader, is_train=False)
        return loop_loss, accuracy

    def loop(self, epochs, train_data):
        """
        Args:
            epochs :        int
                            本地训练轮数

            train_data :    torch.utils.data
                            训练集

        Returns:
            float : loop_loss
                    损失值

            float : accuracy
                    准确率
        """
        epoch_loss, epoch_accuracy = [], []
        for ep in range(1, epochs + 1):
            # if ep < self.epoch + 1:
            #     continue
            loop_loss, accuracy = self.train(train_data)
            epoch_loss.append(loop_loss)
            epoch_accuracy.append(accuracy)
        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def save(self, fpath):
        # if self.save_dir is not None:
        #     state = {"epoch": epoch, "weight": self.model.state_dict()}
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        torch.save(self.model.state_dict(), fpath)

    def restore(self, model_file, include_epoch=False):
        self.say("\n***** restore from {} *****\n".format(model_file))
        model = torch.load(model_file)
        if include_epoch:
            self.epoch = model["epoch"]
        # self.model.load_state_dict(model["weight"])
        self.model.load_state_dict(model)

    def get_lr(self):
        return self.optimizer.state_dict()["param_groups"][0]["lr"]
        # return self.optimizer.param_groups[0]["lr"]

    @property
    def weight(self):
        return self.model.state_dict()

    @property
    def grads(self):
        d = self.weight
        for k, v in d.items():
            d[k] = v - self.model_o[k]
        return d

    def add_grad(self, grads):
        d = self.weight
        for k, v in d.items():
            d[k] = v + grads[k]
        self.model.load_state_dict(d)
