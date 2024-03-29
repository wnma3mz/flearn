# coding: utf-8
import os
from collections import defaultdict
from os.path import join as ospj
from typing import *

from flearn.client.utils import (
    DataInfo,
    bool_key_lst,
    init_log,
    listed_keys,
    str_key_lst,
)
from flearn.common.utils import setup_strategy


class Client(object):
    def __init__(self, conf, **strategy_p) -> None:
        """初始化客户端对象.

        Args:
            conf (dict): {
                "client_id"     :   str,
                                    客户端id

                "epoch"         :   int (default: `1`)
                                    the number of local training rounds, 本地训练轮数

                "trainer"       :   Trainer
                                    训练器

                "trainloader"   :   torch.utils.data
                                    训练数据集

                "testloader"    :   torch.utils.data
                                    测试数据集

                "model_fpath"   :   str
                                    the path to save the model, 本地保存模型的路径, e.g. "/home/"

                "model_name_fmt":   str (default: `"client{}_round_{}.pth".format(self.client_id, "{}")`)
                                    the name of the model, 本地保存模型的名称

                "restore_path"  :   str
                                    restore the path of the trained model, 恢复已经训练模型的路径

                "save_last"     :   bool (default: `False`)
                                    whether to store the latest model, 是否存储最新模型，便于restore

                "save_best"     :   bool (default: `True`)
                                    whether to store the best model, 是否存储最佳模型，便于restore

                "strategy"      :   Strategy
                                    自定义策略

                "scheduler"     :   torch.optim.lr_scheduler
                                    调整学习率, 待调整

                "log"           :   bool (default: `True`)
                                    whether to log client model information, 是否记录客户端log信息

                "log_suffix"    :   str (default: ` `)
                                    log名称的后缀

                "log_name_fmt"  :   str
                                    自定义客户端日志名称
            }
            客户端设置参数

            strategy_p: dict
                        联邦学习策略配置文件。
                        {"shared_key_layers": 共享的参数名称}
        """
        # 设置属性，两种方法均可
        for k, v in conf.items():
            if k in listed_keys:
                self.__dict__[k] = v
                # self.__setattr__(k, v)

        for bool_k in bool_key_lst:
            if bool_k not in self.__dict__.keys():
                self.__dict__[bool_k] = True

        for str_k in str_key_lst:
            if str_k not in self.__dict__.keys():
                self.__dict__[str_k] = None

        if self.model_name_fmt is None:
            self.model_name_fmt = "client{}_round_{}.pth".format(self.client_id, "{}")

        self.fname_fmt = ospj(self.model_fpath, self.model_name_fmt)

        if self.strategy is None:
            self.strategy = setup_strategy(self.strategy_name, None, **strategy_p)

        if self.restore_path != None:
            self.trainer.restore(self.restore_path)

        if self.log == True:
            self.log_client, self.log_fmt = init_log(
                self.log_name_fmt,
                self.client_id,
                self.dataset_name,
                self.log_suffix,
                self.strategy_name,
            )
        if type(self.testloader) != list:
            self.testloader = [self.testloader]

        self.best_acc = 0.0

        name = "client{}_model.pth".format(self.client_id)
        self.update_fpath = ospj(self.model_fpath, name)

        name = "client{}_model_agg.pth".format(self.client_id)
        self.agg_fpath = ospj(self.model_fpath, name)

        name = "client{}_model_best.pth".format(self.client_id)
        self.best_fpath = ospj(self.model_fpath, name)

        # 统计训练数据集的数据信息
        label_distribute_info = defaultdict(lambda: 0)
        for _, y_lst in self.trainloader:
            for y in y_lst:
                label_distribute_info[y.item()] += 1

        num_classes = len(label_distribute_info)
        data_size = sum(label_distribute_info.values())
        data_info = DataInfo(label_distribute_info, num_classes, data_size)
        # 策略可能会利用数据分布信息
        self.strategy.data_info = data_info
        self.trainer.data_info = data_info

    def train(self, i) -> Dict[str, Any]:
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
        self.train_loss, self.train_acc = self.trainer.train(self.trainloader, self.epoch)
        self.upload_model = self.strategy.client(self.trainer, agg_weight=1.0)
        return self._pickle_model()

    def _pickle_model(self) -> Dict[str, Any]:
        if self.save:
            self.trainer.save(self.update_fpath)

        if self.valloader != None:
            _, self.val_acc = self.trainer.test(self.valloader)
        else:
            self.val_acc = -1

        return {
            "code": 200,
            "msg": "Model complete the training",
            "client_id": self.client_id,
            "loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }

    def upload(self, i) -> Dict[str, str]:
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
        model_b64_str = self.strategy.upload_processing(self.upload_model)

        return {
            "code": 200,
            "msg": "Model uploaded successfully",
            "client_id": self.client_id,
            "datas": model_b64_str,
            "fname": os.path.basename(self.fname_fmt.format(i)),
            "round": str(i),
        }

    def receive(self, i, server_p_bytes) -> Dict[str, str]:
        """接收客户端模型.

        Args:
            i : int
                接收第i轮聚合后的模型

        Returns:
            dict: Dict {
                "code"      :   int
                                状态码

                "msg"       :   str
                                状态消息

                "client_id" :   str
                                客户端id

                "round"     :   str
                                轮数
            }

        Notes:
            lr_scheduler 在torch 1.2.0 与 1.4.0中，step(epoch=epoch)表现不一样

            self.scheduler.step(epoch=(i + 1) * self.epoch)
        """
        # update
        self.strategy.client_receive(self.trainer, server_p_bytes)
        if self.scheduler != None:
            self.scheduler.step()

        if self.save:
            self.trainer.save(self.agg_fpath)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }

    def evaluate(self, i) -> Dict[str, str]:
        # Consider the case where there are multiple testloader
        test_acc_lst = [self.trainer.test(loader)[1] for loader in self.testloader]

        # Save the best model against the test results of the first testloader
        if self.save_best and self.best_acc < test_acc_lst[0]:
            self.trainer.save(self.best_fpath)
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
