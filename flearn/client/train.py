# -*- coding: utf-8 -*-

import copy
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import wraps

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def show_f():
    """显示训练/测试过程"""

    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, loader):
            if self.display == True:
                with tqdm(loader, ncols=80, postfix="loss: *.****; acc: *.**") as t:
                    return fn(self, t)
            return fn(self, loader)

        return wrapper

    return inner_fn


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
        self.is_train = None

    def fed_loss(self):
        """联邦学习中，客户端可能需要自定义其他的损失函数"""
        return 0

    @staticmethod
    def metrics(output, target):
        return (output.data.max(1)[1] == target.data).sum().item() / len(target) * 100

    def batch(self, data, target):
        """训练/测试每个batch的数据

        Args:
            data   : torch.tensor
                     训练数据

            target : torch.tensor
                     训练标签

        Returns:
            float : iter_loss
                    对应batch的loss

            float : iter_acc
                    对应batch的accuracy
        """
        output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    @show_f()
    def _iteration(self, loader):
        """模型训练/测试的入口函数, 控制输出显示

        Args:
            data_loader : torch.utils.data
                          数据集

        Returns:
            float :
                    每个epoch的loss取平均

            float :
                    每个epoch的accuracy取平均
        """
        loop_loss, loop_accuracy = [], []
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            iter_loss, iter_acc = self.batch(data, target)
            loop_accuracy.append(iter_acc)
            loop_loss.append(iter_loss)

            if self.display:
                loader.postfix = "loss: {:.4f}; acc: {:.2f}".format(iter_loss, iter_acc)
        return np.mean(loop_loss), np.mean(loop_accuracy)

    def train(self, data_loader):
        """模型训练的入口
        Args:
            data_loader :  torch.utils.data
                           训练集

        Returns:
            float : loss
                    损失值

            float : accuracy
                    准确率
        """
        self.model.train()
        self.is_train = True
        with torch.enable_grad():
            loss, accuracy = self._iteration(data_loader)
        return loss, accuracy

    def test(self, data_loader):
        """模型测试的初始入口，由于只有一轮，所以不需要loop
        Args:
            data_loader :  torch.utils.data
                           测试集

        Returns:
            float : loss
                    损失值

            float : accuracy
                    准确率
        """
        self.model.eval()
        self.is_train = False
        with torch.no_grad():
            loss, accuracy = self._iteration(data_loader)
        return loss, accuracy

    def loop(self, epochs, train_data):
        """模型训练的初始入口，由于可能存在多轮，所以额外套一层函数，便于其他操作
        Args:
            epochs :        int
                            本地训练轮数

            train_data :    torch.utils.data
                            训练集

        Returns:
            float :
                    N个epoch的loss取平均

            float :
                    N个epoch的accuracy取平均
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
        """当前模型的参数"""
        return self.model.state_dict()

    @property
    def grads(self):
        """当前模型的梯度"""
        d = self.weight
        for k, v in d.items():
            d[k] = v - self.model_o[k]
        return d

    def add_grad(self, grads):
        """权重更新"""
        d = self.weight
        for k, v in d.items():
            d[k] = v + grads[k]
        self.model.load_state_dict(d)
