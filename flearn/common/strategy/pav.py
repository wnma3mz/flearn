# coding: utf-8
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .lg_reverse import LG_R


class Distiller:
    def __init__(self, input_len, kd_loader, device, regularization=True):
        # input_len: 512
        self.input_len = input_len
        self.kd_loader = kd_loader
        self.device = device
        self.regularization = regularization

    def kd_generate_soft_label(self, model, data):
        """knowledge distillation (kd): generate soft labels."""
        with torch.no_grad():
            result = model(data)
        if self.regularization:
            # 对输出进行标准化
            result = F.normalize(result, dim=1, p=2)
        return result

    def run(self, teacher_lst, student):
        """
        teacher_lst: 客户端上传的模型参数
        student： 聚合后的模型
        kd_loader: 公开的大数据集
        注：这里的模型没有全连接层
        以每个客户端模型生成的label（平均）来教聚合后的模型
        """
        for teacher in teacher_lst:
            teacher.eval()
            teacher.to(self.device)
        student.train()
        student.to(self.device)
        MSEloss = nn.MSELoss().to(self.device)
        # lr=self.lr*0.01
        optimizer = optim.SGD(
            student.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )

        # kd_loader 公开的大数据集
        for _, (x, target) in enumerate(self.kd_loader):
            x, target = x.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            # 对应于模型全连接层的前一部分，512x10 or 512x100
            soft_target = torch.Tensor([[0] * self.input_len] * len(x)).to(self.device)

            for teacher in teacher_lst:
                soft_label = self.kd_generate_soft_label(teacher, x)
                soft_target += soft_label
            soft_target /= len(teacher_lst)

            output = student(x)

            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            # print("train_loss_fine_tuning", loss.data)
        return student


class PAV(LG_R):
    """
    FedPAV, https://github.com/cap-ntu/FedReID

    note:
    1. soft label or cosine distance
    2. Knowledge Distiller

    References
    ----------
    .. [1] Zhuang W, Wen Y, Zhang X, et al. Performance Optimization of Federated Person Re-identification via Benchmark Analysis[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 955-963.
    """

    def client(self, trainer, old_model, device, trainloader):
        """客户端发送参数

        Args:
            trainer :      Trainer
                                 客户端的训练器

            old_model :          Model
                                 客户端训练前的模型（上一轮更新后的模型）

            device :
                                 训练使用GPU or CPU

            trainloader :        数据集
                                 本地的训练集，仅使用一轮

        Returns:
            dict : Dict {
                'params' :      collections.OrderedDict
                                上传的模型参数，model.state_dict()

                'agg_weight' :  float
                                模型参数所占权重（该客户端聚合所占权重）
            }
        """
        distance = self.cdw_feature_distance(
            old_model, trainer.model, device, trainloader
        )
        # 权重值太小，x100处理
        w_shared = {"agg_weight": np.float(distance) * 100}
        w_local = trainer.weight
        w_shared["params"] = {
            k: v.cpu() for k, v in w_local.items() if k not in self.shared_key_layers
        }
        return w_shared

    @staticmethod
    def load_model(glob_model, w_dict):
        glob_model_dict = glob_model.state_dict()
        glob_model_dict.update(w_dict)
        glob_model.load_state_dict(glob_model_dict)
        return glob_model

    def pav_kd(self, w_local_lst, w_glob, **kwargs):
        # 进行蒸馏
        if "kd" in kwargs.keys() and kwargs["kd"] == True:
            self.distiller = Distiller(
                kwargs["input_len"],
                kwargs["kd_loader"],
                kwargs["device"],
                kwargs["regularization"],
            )

            # 全局模型不完整，所以手动拼接Sequential
            glob_model = kwargs["glob_model"]

            client_lst = []
            # 客户端参数转模型
            for w_local in w_local_lst:
                glob_local = self.load_model(glob_model.state_dict(), w_local)
                glob_model.load_state_dict(glob_local)
                client_lst.append(copy.deepcopy(glob_model))

            glob_local = self.load_model(glob_model.state_dict(), w_glob)
            glob_model.load_state_dict(glob_local)
            # 知识蒸馏+正则化
            glob_model = self.distiller.run(client_lst, glob_model)
            # 模型转回参数
            glob_model_d = glob_model.state_dict()
            for k in w_glob.keys():
                w_glob[k] = glob_model_d[k].cpu()

        return w_glob

    def server_post_processing(self, ensemble_params_lst, ensemble_params, **kwargs):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        ensemble_params["w_glob"] = self.pav_kd(
            w_local_lst, ensemble_params["w_glob"], **kwargs
        )

        return ensemble_params

    def server(self, ensemble_params_lst, round_, **kwargs):
        """服务端聚合客户端模型并蒸馏
        Args:
            kwargs :            dict
                                蒸馏所需参数
        """
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(
            ensemble_params_lst, ensemble_params, **kwargs
        )
