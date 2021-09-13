# coding: utf-8

import base64
import copy
import pickle

import torch.nn as nn
from flearn.client import Client
from flearn.server import Server


class PAVServer(Server):
    def ensemble(self, data_lst, round_, k=-1, **kwargs):
        agg_weight_lst = []
        w_local_lst = []
        client_id_lst = []
        for item in data_lst:
            if int(round_) != int(item["round"]):
                continue
            model_params_encode = item["datas"].encode()
            model_params_b = base64.b64decode(model_params_encode)
            model_data = pickle.loads(model_params_b)

            client_id_lst.append(item["client_id"])
            agg_weight_lst.append(model_data["agg_weight"])
            w_local_lst.append(model_data["params"])

        return self.strategy.server(agg_weight_lst, w_local_lst, round_, **kwargs)


class PAVClient(Client):
    """FedPAV"""

    def __init__(self, conf):
        super(PAVClient, self).__init__(conf)
        assert self.strategy_name.lower() in ["pav"]

    def train(self, i):
        # 注：写死了全连接层
        # if type(self.model_trainer.model) == nn.DataParallel:
        #     old_classifier = copy.deepcopy(self.model_trainer.model.module.fc)
        # else:
        #     old_classifier = copy.deepcopy(self.model_trainer.model.fc)
        # 在原项目中存在old_classifier这个变量，但这里已经合并进old_model中
        old_model = copy.deepcopy(self.model_trainer.model)
        self.train_loss, self.train_acc = self.model_trainer.loop(
            self.epoch, self.trainloader
        )
        w_upload = self.strategy.client(
            self.model_trainer, old_model, self.device, self.trainloader
        )
        
        return super(PAVClient, self)._pickle_model(w_upload)
