# coding: utf-8
import copy

import torch
import torch.nn.functional as F

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.strategy import AVG


class Dyn(AVG):
    def __init__(self, model_fpath, h):
        super(Dyn, self).__init__(model_fpath)
        self.h = h
        self.theta = copy.deepcopy(self.h)
        self.alpha = 0.01

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

        delta_theta = {}
        # assume agg_weight_lst all is 1.0
        for k in self.h.keys():
            delta_theta[k] = w_glob[k] * len(w_local_lst) - self.theta[k]

        for k in self.h.keys():
            self.h[k] -= self.alpha / len(w_local_lst) * delta_theta[k]

        for k in self.h.keys():
            w_glob[k] = w_glob[k] - self.alpha * self.h[k]
        self.theta = w_glob

        return {"w_glob": w_glob}


class DynTrainer(Trainer):
    """
    FedDyn

    References
    ----------
    .. [1] Acar D A E, Zhao Y, Matas R, et al. Federated learning based on dynamic regularization[C]//International Conference on Learning Representations. 2020.

    项目地址: https://github.com/AntixK/FedDyn/blob/17e42576880f5dbe9d7d47cf112f2d88760dca55/feddyn/_feddyn.py#L184-L202
    """

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(DynTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_model = copy.deepcopy(self.model)
        self.server_state_dict = self.server_model.state_dict()

        # save client's gradient
        self.prev_grads = None
        for param in self.model.parameters():
            zero_grad = torch.zeros_like(param.view(-1))
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = zero_grad
            else:
                self.prev_grads = torch.cat((self.prev_grads, zero_grad), dim=0)

        self.alpha = 0.01

    def fed_loss(self):
        if self.server_model != None:
            # Linear penalty
            curr_params = None
            for name, param in self.model.named_parameters():
                if not isinstance(curr_params, torch.Tensor):
                    curr_params = param.view(-1)
                else:
                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)
            # 权重x梯度，尽可能大
            lin_penalty = torch.sum(curr_params * self.prev_grads)

            # Quadratic Penalty, 全局模型与客户端模型尽可能小
            quad_penalty = 0.0
            for name, param in self.model.named_parameters():
                quad_penalty += F.mse_loss(
                    param, self.server_state_dict[name], reduction="sum"
                )

            return -lin_penalty + self.alpha / 2.0 * quad_penalty
        else:
            return 0

    def batch(self, data, target):
        output = self.model(data)
        loss = self.criterion(output, target)

        if self.model.training:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update prev_grads
            self.prev_grads = None
            for param in self.model.parameters():
                real_grad = param.grad.view(-1).clone()
                if not isinstance(self.prev_grads, torch.Tensor):
                    self.prev_grads = real_grad
                else:
                    self.prev_grads = torch.cat((self.prev_grads, real_grad), dim=0)

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class DynClient(Client):
    def revice(self, i, glob_params):
        w_local = self.trainer.weight
        self.w_local_bak = copy.deepcopy(w_local)

        data_glob_d = self.strategy.revice_processing(glob_params)
        # update
        update_w = self.strategy.client_revice(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        # self.trainer.model.load_state_dict(self.w_local_bak)
        self.trainer.model.load_state_dict(update_w)
        self.trainer.server_model = copy.deepcopy(self.self.trainer.model)
        self.trainer.server_model.load_state_dict(update_w)
        self.trainer.server_model.eval()
        self.trainer.server_state_dict = copy.deepcopy(update_w)
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
