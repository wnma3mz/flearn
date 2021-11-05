# coding: utf-8

import base64
import copy
import os
import pickle
from os.path import join as ospj

import torch
import torch.nn as nn

from flearn.common import Encrypt, Logger
from flearn.common.utils import init_strategy
from .Trainer import Trainer
from .utils import bool_key_lst, listed_keys, str_key_lst


class Client(object):
    def __init__(self, conf):
        """初始化客户端对象.

        Args:
            conf (dict): {
                "model" :           torchvision.models
                                    模型,

                "criterion" :       torch.nn.modules.loss
                                    损失函数

                "optimizer" :       torch.optim
                                    优化器

                "trainloader" :     torch.utils.data
                                    训练数据集

                "testloader" :      torch.utils.data
                                    测试数据集

                "device" :          torch.device
                                    使用GPU还是CPU,

                "scheduler" :       torch.optim.lr_scheduler
                                    调整学习率

                "epoch" :           int (default: 1)
                                    本地训练轮数

                "client_id" :       str or int,
                                    客户端id

                "model_fpath" :     str
                                    本地保存模型的路径, /home/

                "model_fname" :     str
                                    本地保存模型的名称, "client{}.pth"

                "save" :            bool
                                    是否存储最新模型，便于restore。(default: `True`)

                "display" :         bool (default: `True`)
                                    是否显示训练过程

                "log" :             bool (default: `True`)
                                    是否记录客户端log信息

                "log_suffix" :      str
                                    log名称的后缀, ""

                "restore_path" :    str
                                    恢复已经训练模型的路径,

                "trainer" :         Trainer
                                    自定义训练器,

                "strategy" :        Strategy
                                    自定义策略

                "log_name_fmt" :    str
                                    自定义客户端日志名称
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

        if self.strategy == None:
            self.strategy = init_strategy(
                self.strategy_name, self.model_fpath, self.shared_key_layers
            )

        if self.restore_path != None:
            self.model.load_state_dict(torch.load(self.restore_path))

        self.trainer = Trainer if self.trainer == None else self.trainer

        self.model_trainer = self.trainer(
            self.model, self.optimizer, self.criterion, self.device, self.display
        )

        if self.log == True:
            self.init_log(self.log_name_fmt)

        if type(self.testloader) != list:
            self.testloader = [self.testloader]

        self.best_acc = 0.0
        self.encrypt = Encrypt()

        name = "client{}_model.pth".format(self.client_id)
        self.update_fpath = ospj(self.model_fpath, name)

        name = "client{}_model_agg.pth".format(self.client_id)
        self.agg_fpath = ospj(self.model_fpath, name)

        name = "client{}_model_best.pth".format(self.client_id)
        self.best_fpath = ospj(self.model_fpath, name)

    def init_log(self, log_name_fmt):
        if log_name_fmt == None:
            log_name_fmt = "{}_client{}_dataset_{}{}.log"
        if self.log_suffix == None:
            self.log_suffix = ""

        log_client_name = log_name_fmt.format(
            self.strategy_name, self.client_id, self.dataset_name, self.log_suffix
        )
        self.log_client = Logger(log_client_name, level="info")
        self.log_fmt = (
            "Id: {}".format(self.client_id)
            + "; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f}; TestAcc: {};"
        )

    def train(self, i):
        """训练客户端模型.

        Args:
            i : int
                进行到第i轮

        Returns:
            dict : Dict {
                "code" :      int
                              状态码,

                "msg" :       str
                              状态消息,

                "loss" :      float
                              训练损失,

                "train_acc" : float
                              模型在训练集上的精度
            }
        """
        self.train_loss, self.train_acc = self.model_trainer.train(
            self.trainloader, self.epoch
        )
        data_upload = self.strategy.client(self.model_trainer, agg_weight=1.0)
        return self._pickle_model(data_upload)

    def _pickle_model(self, data_upload):
        # save，最新的客户端模型
        if self.save:
            self.model_trainer.save(self.update_fpath)

        if self.valloader != None:
            _, self.val_acc = self.model_trainer.test(self.valloader)
        else:
            self.val_acc = -1

        self.upload_model = pickle.dumps(data_upload)

        return {
            "code": 200,
            "msg": "Model complete the training",
            "client_id": self.client_id,
            "loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }

    def upload(self, i):
        """上传客户端模型.

        Args:
            i : int
                上传第i轮训练后的模型

        Returns:
            dict: Dict {
                "datas" : str
                          经过编码（加密）后的模型字符串

                "fname" : str
                          模型名称

                "round" : str
                          第i轮模型

                "msg" :   str
                          状态消息
            }
        """
        # weight params to string
        model_b64_str = self.encrypt.encode(self.upload_model)
        return {
            "code": 200,
            "msg": "Model uploaded successfully",
            "client_id": self.client_id,
            "datas": model_b64_str,
            "fname": os.path.basename(self.fname_fmt.format(i)),
            "round": str(i),
        }

    def revice(self, i, glob_params):
        """接收客户端模型.

        Args:
            i : int
                接收第i轮聚合后的模型

        Returns:
            dict: Dict {
                "code" :     int
                             状态码,

                "msg" :      str
                             状态消息,

                "test_acc" : float
                             模型在测试集上的精度
            }

        Notes:
            lr_scheduler 在torch 1.2.0 与 1.4.0中，step(epoch=epoch)表现不一样

            self.scheduler.step(epoch=(i + 1) * self.epoch)
        """
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

    def evaluate(self, i):
        test_acc_lst = [
            self.model_trainer.test(loader)[1] for loader in self.testloader
        ]

        if self.save:
            self.model_trainer.save(self.agg_fpath)

        if self.best_acc < test_acc_lst[0]:
            self.model_trainer.save(self.best_fpath)
            self.best_acc = test_acc_lst[0]

        if self.log == True and "train_loss" in self.__dict__.keys():
            test_acc_str = "; ".join("{:.4f}".format(x) for x in test_acc_lst)
            log_i = i, self.train_loss, self.train_acc, test_acc_str
            self.log_client.logger.info(self.log_fmt.format(*log_i))
            del self.__dict__["train_loss"]
        return {
            "code": 200,
            "msg": "Model evaluate completed",
            "client_id": self.client_id,
            "test_acc": test_acc_lst,
            "round": str(i),
        }
