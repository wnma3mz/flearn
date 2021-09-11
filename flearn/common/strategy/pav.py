# coding: utf-8
import copy
import pickle
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .strategy import Strategy


class Distillation:
    @staticmethod
    def kd_generate_soft_label(model, data, regularization=True):
        """knowledge distillation (kd): generate soft labels."""
        result = model(data)
        if regularization:
            # 对输出进行标准化
            result = F.normalize(result, dim=1, p=2)
        return result

    def run(self, teacher_lst, student, kd_loader, device, regularization=True):
        """
        teacher_lst: 客户端上传的模型参数
        student： 聚合后的模型
        kd_loader: 公开的大数据集
        注：这里的模型没有全连接层
        以每个客户端模型生成的label（平均）来教聚合后的模型
        """
        for teacher in teacher_lst:
            teacher.eval()
            teacher.to(device)
        student.train()
        student.to(device)
        MSEloss = nn.MSELoss().to(device)
        # lr=self.lr*0.01
        optimizer = optim.SGD(
            student.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )

        # kd_loader 公开的大数据集
        for _, (x, target) in enumerate(kd_loader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            # 对应于模型全连接层的前一部分，512x10 or 512x100
            soft_target = torch.Tensor([[0] * 512] * len(x)).to(device)

            for teacher in teacher_lst:
                soft_label = self.kd_generate_soft_label(teacher, x, regularization)
                soft_target += soft_label
            soft_target /= len(teacher_lst)

            output = student(x)

            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            # print("train_loss_fine_tuning", loss.data)
        return student


class PAV(Strategy):
    """
    FedPAV, https://github.com/cap-ntu/FedReID

    note:
    1. soft label or cosine distance
    2. Knowledge Distillation

    References
    ----------
    .. [1] Zhuang W, Wen Y, Zhang X, et al. Performance Optimization of Federated Person Re-identification via Benchmark Analysis[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 955-963.
    """

    def __init__(self, model_fpath, shared_key_layers, device=None):
        super(PAV, self).__init__(model_fpath)
        self.shared_key_layers = shared_key_layers
        self.device = device
        self.distiller = Distillation()

    def client(
        self, model_trainer, old_model, old_classifier, model, device, trainloader
    ):
        """
        发送参数前调用
        old_model，全局模型，聚合后的模型，
        old_classifier，原来的全连接层
        model，训练后的模型
        """
        w_local = model_trainer.weight
        distance = self.cdw_feature_distance(
            old_model, old_classifier, model, device, trainloader
        )
        w_shared = {"params": {}, "agg_weight": np.float(distance)}
        for k in w_local.keys():
            if k not in self.shared_key_layers:
                w_shared["params"][k] = w_local[k].cpu()
        return w_shared

    def client_revice(self, model_trainer, w_glob_b):
        w_local = model_trainer.weight
        w_glob = pickle.loads(w_glob_b)
        for k in w_glob.keys():
            if k not in self.shared_key_layers:
                w_local[k] = w_glob[k]
        model_trainer.model.load_state_dict(w_local)
        return model_trainer.model

    @staticmethod
    def load_model(glob_local, w_glob):
        for k in glob_local.keys():
            if k in w_glob.keys():
                glob_local[k] = w_glob[k]
        return glob_local

    def server(self, agg_weight_lst, w_local_lst, round_, **kwargs):
        try:
            # 取公共的key
            key_lst = reduce(lambda x, y: set(x.keys()) & set(y.keys()), w_local_lst)
            key_lst = [k for k in key_lst if k not in self.shared_key_layers]

            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        except Exception as e:
            return self.server_exception(e)

        if "kd" in kwargs.keys() and kwargs["kd"] == True:
            glob_model = kwargs["glob_model"]
            glob_model.fc = nn.Sequential()

            client_lst = []
            # 客户端参数转模型
            for w_local in w_local_lst:
                glob_local = self.load_model(glob_model.state_dict(), w_local)
                glob_model.load_state_dict(glob_local)
                client_lst.append(copy.deepcopy(glob_model))

            glob_local = self.load_model(glob_model.state_dict(), w_glob)
            glob_model.load_state_dict(glob_local)
            # 知识蒸馏+正则化
            glob_model = self.distiller.run(
                client_lst,
                glob_model,
                kwargs["kd_loader"],
                kwargs["device"],
                regularization=kwargs["regularization"],
            )
            # 模型转回参数
            glob_model_d = glob_model.state_dict()
            for k in w_glob.keys():
                w_glob[k] = glob_model_d[k].cpu()

        return self.server_post_processing(w_glob, round_)
