# coding: utf-8
from os.path import join as ospj

from flearn.client.utils import bool_key_lst, init_log, listed_keys, str_key_lst


class DLClient(object):
    """单独训练完整数据集"""

    def __init__(self, conf):
        """初始化客户端对象.

        Args:
            conf (dict): {
                "client_id" :       str or int,
                                    客户端id

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

                "trainer" :         Trainer
                                    训练器

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
            self.trainer.restore(self.restore_path)

        if self.log == True:
            self.log_client, self.log_fmt = init_log(
                self.log_name_fmt, self.client_id, self.dataset_name, self.log_suffix
            )
        self.best_acc = 0.0
        self.update_fpath = ospj(
            self.model_fpath, "client{}_model.pth".format(self.client_id)
        )

        self.best_fpath = ospj(
            self.model_fpath, "client{}_model_best.pth".format(self.client_id)
        )

    def run(self, i, test_freq=1):
        """训练客户端模型.

        Args:
            i        :  int
                        进行到第i轮

            test_freq:  int
                        间隔print_freq轮进行测试
        """

        train_loss, train_acc = self.trainer.train(self.trainloader)

        # save，最新的客户端模型
        test_log = ""
        if i % test_freq == 0:
            _, test_acc = self.trainer.test(self.testloader)
            test_log = " TestAcc: {:.4f};".format(test_acc)

        if self.scheduler:
            self.scheduler.step()

        if self.best_acc > test_acc:
            self.trainer.save(self.best_fpath)
            self.best_acc = test_acc

        log_i = i, train_loss, train_acc, test_log
        if self.log == True:
            self.log_client.logger.info(self.log_fmt.format(*log_i))
        return log_i
