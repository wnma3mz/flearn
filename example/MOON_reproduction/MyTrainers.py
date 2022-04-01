# coding: utf-8

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from flearn.common import Trainer
from flearn.common.distiller import KDLoss


class AVGTrainer(Trainer):
    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


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

    def fed_loss(self):
        if self.global_model != None:
            data, pro1 = self.data, self.pro1
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

    def eval_model(self):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, pro1, output = self.model(data)
        self.data, self.pro1 = data, pro1

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class ProxTrainer(AVGTrainer):
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

    def eval_model(self):
        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)


class DynTrainer(AVGTrainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(DynTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_state_dict = {}

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
        if self.server_state_dict != {}:
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
                server_param = self.server_state_dict[name].to(self.device)
                quad_penalty += F.mse_loss(param, server_param, reduction="sum")

            return -lin_penalty + self.alpha / 2.0 * quad_penalty
        else:
            return 0

    def update_info(self):
        # update prev_grads
        self.prev_grads = None
        for param in self.model.parameters():
            real_grad = param.grad.view(-1).clone()
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = real_grad
            else:
                self.prev_grads = torch.cat((self.prev_grads, real_grad), dim=0)


class LSDTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(LSDTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.teacher_model = None
        self.kd_mu = 2
        # self.kd_mu = 0.5
        self.kd_loss = KDLoss(2)

    def eval_model(self):
        if self.teacher_model != None:
            self.teacher_model.eval()
            self.teacher_model.to(self.device)

    def train(self, data_loader, epochs=1):
        self.lsd_eval_model()
        return super(LSDTrainer, self).train(data_loader, epochs)

    def fed_loss(self):
        if self.teacher_model != None:
            data, output = self.data, self.output
            with torch.no_grad():
                t_h, _, t_output = self.teacher_model(data)
            return self.kd_mu * self.kd_loss(output, t_output.detach())
        return 0

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)
        self.data, self.output = data, output

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class LogitTracker:
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.clear()

    def clear(self):
        self.labels = [i for i in range(self.unique_labels)]
        # avoid division by zero error
        self.label_counts = torch.ones(self.unique_labels)
        self.logit_sums = torch.zeros((self.unique_labels, self.unique_labels))

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
        super().__init__(model, optimizer, criterion, device, display)
        self.logit_tracker = LogitTracker(10)  # cifar10
        self.glob_logit = None
        self.kd_mu = 1
        self.kd_loss = KDLoss(2)

    def fed_loss(self):
        if self.glob_logit != None:
            output, target = self.output, self.target
            self.glob_logit = self.glob_logit.to(self.device)
            target_p = self.glob_logit[target, :]
            return self.kd_mu * self.kd_loss(output, target_p)
        return 0

    def update_info(self):
        # 更新上传的logits
        self.logit_tracker.update(self.output, self.target)

    def clear_info(self):
        self.logit_tracker.clear()

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)
        self.output, self.target = output, target.to(self.device)

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class CCVRTrainer(AVGTrainer):
    # 从左至右继承，右侧不会覆盖左侧的变量/函数
    def __init__(self, base_trainer):
        super().__init__(
            base_trainer.model,
            base_trainer.optimizer,
            base_trainer.criterion,
            base_trainer.device,
            base_trainer.display,
        )
        self.feat_lst, self.label_lst = [], []
        self.base_trainer = base_trainer
        for k, v in base_trainer.__dict__.items():
            if k not in self.__dict__.keys():
                self.__dict__[k] = v
        self.eval_model = self.base_trainer.eval_model

    def update_info(self):
        # 保存中间特征
        h, target = self.h, self.target
        self.feat_lst.append(h)
        self.label_lst.append(target)
        self.base_trainer.update_info()

    def clear_info(self):
        self.feat_lst, self.label_lst = [], []
        self.base_trainer.clear_info()

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        h, pro1, output = self.model(data)

        # 更新所有可能需要的数据
        self.h, self.data, self.pro1 = h, data, pro1
        self.output, self.target = output, target
        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc
