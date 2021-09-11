# coding: utf-8

import base64
import os
import pickle
from os.path import join as ospj

import torch
import torch.nn as nn

from flearn.common import Logger
from .train import Trainer
from .utils import bool_key_lst, listed_keys, str_key_lst


class DLClient(object):
    """单独训练完整数据集"""

    def __init__(self, conf):
        """初始化客户端对象.

        Args:
            conf (dict): {
                "model" :           torchvision.models
                                    模型,

                "epoch" :           int (default: 1)
                                    本地训练轮数

                "client_id" :       str or int,
                                    客户端id

                "criterion" :       torch.nn.modules.loss
                                    损失函数

                "optimizer" :       torch.optim
                                    优化器

                "trainloader" :     torch.utils.data
                                    训练数据集

                "testloader" :      torch.utils.data
                                    测试数据集

                "model_fpath" :     str
                                    本地保存模型的路径, /home/

                "model_fname" :     str
                                    本地保存模型的名称, "client{}.pth"

                "save" :            bool
                                    是否存储最新模型，便于restore。(default: `True`)

                "display" :         bool (default: `True`)
                                    是否显示训练过程

                "restore_path" :    str
                                    恢复已经训练模型的路径,

                "custom_trainer" :  object
                                    自定义训练器,

                "device" :          torch.device
                                    使用GPU还是CPU,

                "scheduler" :       torch.optim.lr_scheduler
                                    调整学习率
            }
            客户端设置参数
        """
        # 设置属性，两种方法均可
        for k in conf.keys():
            if k in listed_keys:
                self.__dict__[k] = conf[k]
                # self.__setattr__(k, kwargs[k])

        for bool_k in bool_key_lst:
            if bool_k not in self.__dict__.keys():
                self.__dict__[bool_k] = True

        for str_k in str_key_lst:
            if str_k not in self.__dict__.keys():
                self.__dict__[str_k] = None

        self.fname_fmt = ospj(self.model_fpath, self.model_fname)

        if self.restore_path:
            self.model.load_state_dict(torch.load(self.restore_path))

        if self.trainer == None:
            self.trainer = Trainer

        self.model_trainer = self.trainer(
            self.model, self.optimizer, self.criterion, self.device, self.display
        )

        if self.log == True:
            self.init_log(self.log_name_fmt)
        self.best_acc = 0.0
        self.update_fpath = ospj(
            self.model_fpath, "client{}_model.pth".format(self.client_id)
        )

        self.best_fpath = ospj(
            self.model_fpath, "client{}_model_best.pth".format(self.client_id)
        )

    def init_log(self, log_name_fmt):
        if log_name_fmt == None:
            log_name_fmt = "client{}_dataset_{}{}.log"
        if self.log_suffix == None:
            self.log_suffix = ""

        log_client_name = log_name_fmt.format(
            self.client_id,
            self.dataset_name,
            self.log_suffix,
        )
        self.log_client = Logger(log_client_name, level="info")
        self.log_fmt = (
            "Id: {}".format(self.client_id)
            + "; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f}; TestAcc: {:.4f};"
        )

    def run(self, i):
        """训练客户端模型.

        Args:
            i : int
                进行到第i轮
        """

        train_loss, train_acc = self.model_trainer.loop(self.epoch, self.trainloader)

        # save，最新的客户端模型
        if self.save:
            self.model_trainer.save(self.update_fpath)
        _, test_acc = self.model_trainer.test(self.testloader)
        if self.scheduler:
            self.scheduler.step()

        if self.best_acc > test_acc:
            self.model_trainer.save(self.best_fpath)
            self.best_acc = test_acc

        log_i = i, train_loss, train_acc, test_acc
        if self.log == True:
            self.log_client.logger.info(self.log_fmt.format(*log_i))
        return log_i
