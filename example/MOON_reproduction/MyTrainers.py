# coding: utf-8

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flearn.common import Trainer
from flearn.common.distiller import KDLoss


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
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(MOONTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = copy.deepcopy(self.model)
        self.previous_model_lst = []
        self.cos = nn.CosineSimilarity(dim=-1)
        # CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.5, 1, and 0.5
        self.temperature = 0.5
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 5, 1, and 1
        self.mu = 5

    def moon_loss(self, data, pro1):
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

    def moon_eval_model(self):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)

    def train(self, data_loader, epochs=1):
        self.moon_eval_model()
        return super(MOONTrainer, self).train(data_loader, epochs)

    def batch(self, data, target):
        _, pro1, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            loss += self.moon_loss(data, pro1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class ProxTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(ProxTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = None
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.01, 0.001, and 0.001
        self.prox_mu = 0.01

    def prox_loss(self):
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
            loss += self.prox_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


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

    def dyn_loss(self):
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

    def update_prev_grads(self):
        # update prev_grads
        self.prev_grads = None
        for param in self.model.parameters():
            real_grad = param.grad.view(-1).clone()
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = real_grad
            else:
                self.prev_grads = torch.cat((self.prev_grads, real_grad), dim=0)

    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.dyn_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_prev_grads()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class LSDTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(LSDTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.teacher_model = None
        self.mu_kd = 2
        # self.mu_kd = 0.5
        self.kd_loss = KDLoss(2)

    def lsd_eval_model(self):
        if self.teacher_model != None:
            self.teacher_model.eval()
            self.teacher_model.to(self.device)

    def train(self, data_loader, epochs=1):
        self.lsd_eval_model()
        return super(LSDTrainer, self).train(data_loader, epochs)

    def lsd_loss(self, data, output):
        if self.teacher_model != None:
            with torch.no_grad():
                t_h, _, t_output = self.teacher_model(data)

            return self.mu_kd * self.kd_loss(output, t_output.detach())
        return 0

    def batch(self, data, target):
        h, _, output = self.model(data)
        loss = self.criterion(output, target)
        if self.is_train:
            loss += self.lsd_loss(data, output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class LogitTracker:
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels)  # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels, unique_labels))

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        logits, Y = logits.to("cpu"), Y.to("cpu")
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels))
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_

    def avg(self):
        return self.logit_sums.detach() / self.label_counts.float().unsqueeze(1)


class DistillTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(DistillTrainer, self).__init__(
            model, optimizer, criterion, device, display
        )
        self.logit_tracker = LogitTracker(10)  # cifar10
        self.glob_logit = None
        self.kd_mu = 1
        self.kd_loss = KDLoss(2)

    def train(self, data_loader, epochs=1):
        self.model.train()
        self.is_train = True
        epoch_loss, epoch_accuracy = [], []
        for ep in range(1, epochs + 1):
            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)

            # 非上传轮，清空特征
            if ep != epochs:
                self.logit_tracker.clear()

        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def distill_loss(self, output, target):
        if self.glob_logit != None:
            self.glob_logit = self.glob_logit.to(self.device)
            target_p = self.glob_logit[target, :]
            return self.kd_mu * self.kd_loss(output, target_p)
        return 0

    def batch(self, data, target):
        _, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.distill_loss(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新上传的logits
            self.logit_tracker.update(output, target)

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class CCVRTrainer(MOONTrainer, ProxTrainer, LSDTrainer, DistillTrainer, DynTrainer):
    # 从左至右继承，右侧不会覆盖左侧的变量/函数
    def __init__(
        self, model, optimizer, criterion, device, display=True, strategy=None
    ):
        super(CCVRTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.feat_lst = []
        self.label_lst = []
        self.fed_loss_d = {
            "avg": 0,
            "lg": 0,
            "prox": self.prox_loss(),
            "dyn": self.dyn_loss(),
        }
        assert strategy in ["avg", "moon", "prox", "dyn", "lsd", "distill", "lg"]
        self.strategy = strategy

    def train(self, data_loader, epochs=1):
        self.moon_eval_model()
        self.lsd_eval_model()

        self.model.train()
        self.is_train = True
        epoch_loss, epoch_accuracy = [], []
        for ep in range(1, epochs + 1):
            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)

            # 非上传轮，清空特征
            if ep != epochs:
                self.feat_lst = []
                self.label_lst = []

                if self.strategy == "distill":
                    self.logit_tracker.clear()

        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def update_feat(self, h, target):
        # 保存中间特征
        self.feat_lst.append(h)
        self.label_lst.append(target)

    def batch(self, data, target):
        h, pro1, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            if self.strategy == "lsd":
                loss += self.lsd_loss(data, output)
            elif self.strategy == "distill":
                loss += self.distill_loss(output, target)
            elif self.strategy == "moon":
                loss += self.moon_loss(data, pro1)
            else:
                loss += self.fed_loss_d[self.strategy]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_feat(h, target)
            if self.strategy == "distill":
                self.logit_tracker.update(output, target)
            elif self.strategy == "dyn":
                self.update_prev_grads()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc
