# coding: utf-8
import copy
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flearn.common import Trainer
from flearn.common.strategy import AVG


class AVGTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super().__init__(model, optimizer, criterion, device, display)
        # 源代码的梯度，是-pretrain的值？
        self.model_o = copy.deepcopy(self.model.state_dict())
        self.mse_criterion = nn.MSELoss()
        self.kl_criterion = nn.KLDivLoss()
        self.temp = 1

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        # _, _, output = self.model(data)

        # (th, sh), (tx, sx), (ty, sy) = self.model(data)
        (th_lst, sh_lst), (ty, sy) = self.model(data)

        loss_ce = self.criterion(ty, target) + self.criterion(sy, target)
        # loss_mse = 0.0
        loss_mse = (
            self.mse_criterion(th_lst[-1], sh_lst[-1]) / loss_ce
            + self.mse_criterion(th_lst[-2], sh_lst[-2]) / loss_ce
        )

        def ts_kl_f(a, b):
            a_log_soft = F.log_softmax(a / self.temp, dim=1)
            b_soft = F.softmax(b / self.temp, dim=1)
            return self.kl_criterion(a_log_soft, b_soft)

        loss_kl = ts_kl_f(ty, sy) / loss_ce + ts_kl_f(sy, ty) / loss_ce
        loss = loss_ce + loss_kl + loss_mse

        # 教师输出的精度
        output = ty
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc

    # def train(self, data_loader, epochs=1):
    #     self.model_o = copy.deepcopy(self.model.state_dict())
    #     return super().train(data_loader, epochs)


class FedKD(AVG):
    """
    客户端两个模型的 Loss

    仅上传学生模型（小模型）的参数，且用SVD后的参数进行传输
    [1]

    学生模型和教师模型分别在model中实现
    """

    # def client(self, trainer, agg_weight=1):
    #     w_shared = {"agg_weight": agg_weight}
    #     w_local = trainer.weight
    #     w_shared["params"] = {
    #         k: v.cpu() for k, v in w_local.items() if "teacher" not in k
    #     }
    #     return w_shared

    # https://github.com/wuch15/FedKD/blob/main/run.py
    def client(self, trainer, agg_weight=1):
        # 随着轮数的变化而变化， svd的k, energy
        # energy = 0.95+((1+comm_round)/10)*(0.98-0.95)
        self.energy = 1  # init_value
        w_shared = {"agg_weight": agg_weight}
        # w_local = trainer.weight
        w_local = trainer.grads
        w_shared["params"] = {}
        for key, value in w_local.items():
            conv_flag = False
            params_mat = value.cpu().numpy()
            w_shared["params"][key] = params_mat
            if "bias" not in key and len(params_mat.shape) > 1:
                # 卷积层
                if len(params_mat.shape) == 4:
                    conv_flag = True
                    c, k, h, w = params_mat.shape
                    params_mat = params_mat.reshape(c * k, h * w)

                U, Sigma, VT = np.linalg.svd(params_mat, full_matrices=False)

                threshold = 0
                sigma_square_sum = np.sum(np.square(Sigma))
                if sigma_square_sum != 0:
                    for singular_value_num in range(len(Sigma)):
                        if (
                            np.sum(np.square(Sigma[:singular_value_num]))
                            > self.energy * sigma_square_sum
                        ):
                            threshold = singular_value_num
                            break
                    U = U[:, :threshold]
                    Sigma = Sigma[:threshold]
                    VT = VT[:threshold, :]

                    # 原代码是在服务器上进行dot，但这样会增加通信成本（需要传输u、sigma、v），所以这里换成本地实现
                    # con_restruct1 = np.dot(np.dot(U, np.diag(Sigma)), VT)
                    w_shared["params"][key] = np.dot(np.dot(U, np.diag(Sigma)), VT)
                    if conv_flag:
                        w_shared["params"][key] = w_shared["params"][key].reshape(
                            c, k, h, w
                        )
        return w_shared

    def server_ensemble(self, agg_weight_lst, w_local_lst, key_lst=None):
        if key_lst == None:
            all_local_key_lst = [set(w_local.keys()) for w_local in w_local_lst]
            key_lst = reduce(lambda x, y: x & y, all_local_key_lst)
        # sum up weights
        w_glob = {k: agg_weight_lst[0] * w_local_lst[0][k] for k in key_lst}
        for agg_weight, w_local in zip(agg_weight_lst[1:], w_local_lst[1:]):
            for k in key_lst:
                w_glob[k] += agg_weight * w_local[k]
        molecular = np.sum(agg_weight_lst)
        for k in w_glob.keys():
            w_glob[k] = np.divide(w_glob[k], molecular)
        return w_glob

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight_o
        w_glob = data_glob_d["w_glob"]

        for key, value in w_glob.items():
            real_params_value = value
            conv_flag = False

            # 类似的，在服务器端除了要dot，再mean之后还需要再做一次svd。这里换成本地实现
            if "bias" not in key and len(value.shape) > 1:
                # 卷积层
                if len(value.shape) == 4:
                    conv_flag = True
                    c, k, h, w = value.shape
                    params_mat = value.reshape(c * k, h * w)
                else:
                    params_mat = value

                U, Sigma, VT = np.linalg.svd(params_mat, full_matrices=False)
                sigma_square_sum = np.sum(np.square(Sigma))

                if sigma_square_sum != 0:
                    threshold = 0
                    for singular_value_num in range(len(Sigma)):
                        if np.sum(
                            np.square(Sigma[:singular_value_num])
                        ) >= self.energy * np.sum(np.square(Sigma)):
                            threshold = singular_value_num
                            break
                    U = U[:, :threshold]
                    Sigma = Sigma[:threshold]
                    VT = VT[:threshold, :]
                    # t_lst = [u, sigma, v]
                    real_params_value = np.dot(np.dot(U, np.diag(Sigma)), VT)
                    if conv_flag:
                        real_params_value = real_params_value.reshape(c, k, h, w)

            w_local[key] = w_local[key] + torch.FloatTensor(real_params_value)
        return w_local


if __name__ == "__main__":
    from model import ModelFedCon

    model_base = ModelFedCon("simple-cnn", out_dim=256, n_classes=10)
    d = model_base.state_dict()
    conv_m = d["features.conv1.weight"].numpy()
    fc_m = d["l1.weight"].numpy()

    u, s, v = np.linalg.svd(fc_m, full_matrices=False)

    t1_r = np.dot(np.dot(u, np.diag(s)), v)
    t1_dist = torch.dist(torch.tensor(fc_m), torch.tensor(t1_r))
    print(t1_dist)

    t2_r = np.dot(u, np.diag(s), v)
    t2_dist = torch.dist(torch.tensor(fc_m), torch.tensor(t2_r))
    print(t2_dist)

    t3_r = np.matmul(u, np.diag(s), v)
    t3_dist = torch.dist(torch.tensor(fc_m), torch.tensor(t3_r))
    print(t3_dist)

    # u, s, v = np.linalg.svd(conv_m, full_matrices=False)
    U, Sigma, VT = np.linalg.svd(
        np.reshape(
            conv_m,
            (
                conv_m.shape[0] * conv_m.shape[1],
                conv_m.shape[2] * conv_m.shape[3],
            ),
        ),
        full_matrices=False,
    )
    con_restruct1 = np.dot(np.dot(U, np.diag(Sigma)), VT)
    t4_r = np.reshape(
        con_restruct1,
        (conv_m.shape[0], conv_m.shape[1], conv_m.shape[2], conv_m.shape[3]),
    )
    # t4_r = np.dot(np.dot(u, s[:, None, :]), v)
    t4_dist = torch.dist(torch.tensor(conv_m), torch.tensor(t4_r))
    print(t4_dist)
