# coding: utf-8
import copy

from flearn.client import Client


class PAVClient(Client):
    """FedPAV"""

    def __init__(self, conf):
        super(PAVClient, self).__init__(conf)
        assert self.strategy_name.lower() in ["pav"]

    def train(self, i):
        # 注：写死了全连接层
        # if type(self.trainer.model) == nn.DataParallel:
        #     old_classifier = copy.deepcopy(self.trainer.model.module.fc)
        # else:
        #     old_classifier = copy.deepcopy(self.trainer.model.fc)
        # 在原项目中存在old_classifier这个变量，但这里已经合并进old_model中
        old_model = copy.deepcopy(self.trainer.model)
        self.train_loss, self.train_acc = self.trainer.train(self.trainloader, self.epoch)
        w_upload = self.strategy.client(self.trainer, old_model, self.device, self.trainloader)

        return super(PAVClient, self)._pickle_model(w_upload)
