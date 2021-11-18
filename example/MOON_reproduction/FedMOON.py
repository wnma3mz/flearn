# coding: utf-8

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.strategy import AVG


class AVGTrainer(Trainer):
    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class MOONTrainer(Trainer):
    """"搭配MOONClient使用"""

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(MOONTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = copy.deepcopy(self.model)
        self.previous_model_lst = []
        self.cos = nn.CosineSimilarity(dim=-1)
        # CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.5, 1, and 0.5
        self.temperature = 0.5
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 5, 1, and 1
        self.mu = 5

    def fed_loss(self, data, pro1):
        if self.global_model != None:
            # 全局与本地的对比损失，越小越好
            with torch.no_grad():
                _, pro2, _ = self.global_model(data)
            posi = self.cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # 当前轮与上一轮的对比损失，越大越好
            for previous_net in self.previous_model_lst:
                with torch.no_grad():
                    _, pro3, _ = previous_net(data)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= self.temperature
            labels = torch.zeros(data.size(0)).to(self.device).long()

            return self.mu * self.criterion(logits, labels)
        else:
            return 0

    def train(self, data_loader, epochs=1):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)
        return super(MOONTrainer, self).train(data_loader, epochs)

    def batch(self, data, target):
        _, pro1, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            loss += self.fed_loss(data, pro1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class MOONClient(Client):
    """MOONClient"""

    def __init__(self, conf, pre_buffer_size=1):
        super(MOONClient, self).__init__(conf)
        # 保存以前模型的大小
        self.pre_buffer_size = pre_buffer_size
        # 记录运行轮数
        self.ci = -1

    def train(self, i):
        # 每轮训练+1
        self.ci += 1
        self.train_loss, self.train_acc = self.model_trainer.train(
            self.trainloader, self.epoch
        )
        # 权重为本地数据大小
        self.upload_model = self.strategy.client(
            self.model_trainer, agg_weight=len(self.trainloader)
        )
        return self._pickle_model()

    def revice(self, i, glob_params):
        # 额外需要两类模型，glob和previous，一般情况下glob只有一个，previous也定义只有一个
        # 如果存储超过这个大小，则删除最老的模型
        while len(self.model_trainer.previous_model_lst) >= self.pre_buffer_size:
            self.model_trainer.previous_model_lst.pop(0)
        self.model_trainer.previous_model_lst.append(
            copy.deepcopy(self.model_trainer.model)
        )

        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)

        # update
        update_w = self.strategy.client_revice(self.model_trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model.load_state_dict(update_w)

        # 如果该客户端训练轮数不等于服务器端的训练轮数，则表示该客户端的模型本轮没有训练，则不做对比学习，并且同步进度轮数。
        if self.ci != i:
            self.model_trainer.global_model = None
            self.model_trainer.previous_model_lst = []
            self.ci = i - 1
        else:
            self.model_trainer.global_model = copy.deepcopy(self.model_trainer.model)
            self.model_trainer.global_model.load_state_dict(update_w)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class ProxTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(ProxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = None
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.01, 0.001, and 0.001
        self.prox_mu = 0.01

    def fed_loss(self):
        if self.global_model != None:
            w_diff = torch.tensor(0.0, device=self.device)
            for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            return self.prox_mu / 2.0 * w_diff
        else:
            return 0

    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    def train(self, data_loader, epochs=1):
        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)
        return super(ProxTrainer, self).train(data_loader, epochs)


class ProxClient(Client):
    def revice(self, i, glob_params):
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)

        # update
        update_w = self.strategy.client_revice(self.model_trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model.load_state_dict(update_w)
        self.model_trainer.global_model = copy.deepcopy(self.model_trainer.model)
        self.model_trainer.global_model.load_state_dict(update_w)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


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
            # # 权重x梯度，尽可能大
            lin_penalty = torch.sum(curr_params * self.prev_grads)
            # lin_penalty = 0
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
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
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
        w_local = self.model_trainer.weight
        self.w_local_bak = copy.deepcopy(w_local)
        # decode
        data_glob_d = self.strategy.revice_processing(glob_params)
        # update
        update_w = self.strategy.client_revice(self.model_trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        # self.model_trainer.model.load_state_dict(self.w_local_bak)
        self.model_trainer.model.load_state_dict(update_w)
        self.model_trainer.server_model = copy.deepcopy(self.model_trainer.model)
        self.model_trainer.server_model.load_state_dict(update_w)

        self.model_trainer.server_model.eval()
        self.model_trainer.server_state_dict = copy.deepcopy(update_w)
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
