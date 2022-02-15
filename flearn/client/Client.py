# coding: utf-8
import os
from os.path import join as ospj

from flearn.client.utils import bool_key_lst, listed_keys, str_key_lst
from flearn.common import Encrypt, Logger, setup_strategy


class Client(object):
    def __init__(self, conf, **strategy_p):
        """初始化客户端对象.

        Args:
            conf (dict): {
                "client_id" :       str or int,
                                    客户端id

                "epoch" :           int (default: 1)
                                    本地训练轮数

                "trainer" :         Trainer
                                    训练器

                "trainloader" :     torch.utils.data
                                    训练数据集

                "testloader" :      torch.utils.data
                                    测试数据集

                "model_fpath" :     str
                                    本地保存模型的路径, /home/

                "model_fname" :     str
                                    本地保存模型的名称, "client{}.pth"

                "restore_path" :    str
                                    恢复已经训练模型的路径,

                "save" :            bool
                                    是否存储最新模型，便于restore。(default: `True`)

                "strategy" :        Strategy
                                    自定义策略

                "scheduler" :       torch.optim.lr_scheduler
                                    调整学习率, 待调整

                "log" :             bool (default: `True`)
                                    是否记录客户端log信息

                "log_suffix" :      str
                                    log名称的后缀, ""

                "log_name_fmt" :    str
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

        self.fname_fmt = ospj(self.model_fpath, self.model_fname)

        if self.strategy == None:
            self.strategy = setup_strategy(self.strategy_name, None, **strategy_p)

        if self.restore_path != None:
            self.trainer.restore(self.restore_path)

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
            log_name_fmt = "[Client-{}]{}_dataset_{}{}.log"
        if self.log_suffix == None:
            self.log_suffix = ""

        log_client_name = log_name_fmt.format(
            self.client_id, self.strategy_name, self.dataset_name, self.log_suffix
        )
        self.log_client = Logger(log_client_name, level="info")
        self.log_fmt = (
            self.client_id + "; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f}; TestAcc: {};"
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
        self.train_loss, self.train_acc = self.trainer.train(
            self.trainloader, self.epoch
        )
        self.upload_model = self.strategy.client(self.trainer, agg_weight=1.0)
        return self._pickle_model()

    def _pickle_model(self):
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
        model_b64_str = self.strategy.upload_processing(self.upload_model)

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
        data_glob_d = self.strategy.revice_processing(glob_params)

        # update
        w_update = self.strategy.client_revice(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(w_update)

        if self.save:
            self.trainer.save(self.agg_fpath)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }

    def evaluate(self, i):
        # Consider the case where there are multiple testloader
        test_acc_lst = [self.trainer.test(loader)[1] for loader in self.testloader]

        # Save the best model against the test results of the first testloader
        if self.best_acc < test_acc_lst[0]:
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
