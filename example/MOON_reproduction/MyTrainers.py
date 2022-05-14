# coding: utf-8
import torch

from flearn.common.distiller import KDLoss
from flearn.common.trainer import DistillTrainer, DynTrainer, ProxTrainer, Trainer


class AVGTrainer(Trainer):
    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class MyProxTrainer(ProxTrainer):
    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


class MyDynTrainer(DynTrainer):
    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, _, output = self.model(data)

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc


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


class MyDistillTrainer(DistillTrainer):
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
