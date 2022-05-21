# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        q = torch.softmax(target / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q) * (self.temp_factor ** 2) / input.size(0)
        return loss


class DistillLoss(nn.Module):
    def __init__(self, T, alpha):
        super(DistillLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="sum")
        # self.kl_div = F.mse_loss
        # self.kl_div = F.l1_loss

    def forward(self, y, soft_target=None, labels=None, t_probs=None):
        """
        soft_target: None, 教师输出
        labels  : None, 无监督蒸馏, 此时self.alpha=1
        t_probs : None, 不提供教师输出标签（soft_target）
        soft_target与t_probs二者中必须传一个参数
        """
        is_tensor = isinstance(soft_target, torch.Tensor) or isinstance(
            t_probs, torch.Tensor
        )
        if not is_tensor:
            raise SystemError("Please input soft_target or t_probs")

        cross_loss = F.cross_entropy(y, labels)
        s_probs = F.log_softmax(y / self.T, dim=1)
        if type(t_probs) != torch.Tensor:
            t_probs = F.softmax(soft_target / self.T, dim=1)
        kd_loss = self.alpha * self.kl_div(s_probs, t_probs) * self.T * self.T * 2.0

        return kd_loss + (1.0 - self.alpha) * cross_loss


class Distiller:
    def __init__(self, kd_loader, device, kd_loss=DistillLoss(2, 0.5)):
        """知识蒸馏训练器

        Args:
            kd_loader : DataLoader
                        蒸馏训练的数据集，为None，则随机生成

            kd_loss :   nn.Module
                        蒸馏损失函数，默认为KL散度

            device :    str
                        蒸馏训练所在的设备，gpu or cpu

        """
        self.loss = kd_loss
        self.kd_loader = kd_loader
        self.device = device

    def _init_kd(self, teacher, student, **kwargs):
        """初始化蒸馏参数
        Args:
            teacher :
                        教师模型

            student :
                        学生模型

            kwargs :    dict
                        {"lr": 学习率, "epoch": 蒸馏轮数}
        """
        self.lr = kwargs["lr"]
        self.epoch = kwargs["epoch"]

        if type(teacher) == list:
            self.teacher_lst = teacher
            for t in self.teacher_lst:
                t.to(self.device)
                t.eval()
        else:
            self.teacher = teacher
            self.teacher.to(self.device)
            self.teacher.eval()

        self.student = student
        self.student.to(self.device)
        self.student.train()

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.lr
        )

        # self.optimizer = optim.Adam(student.parameters(), lr=lr)
        # self.optimizer = optim.SGD(
        #     student.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True
        # )

    def single(self, teacher, student, **kwargs):
        """经典的知识蒸馏方法

        Args:
            teacher :
                        教师模型

            student :
                        学生模型

            kwargs :    dict
                        {"lr": 学习率, "epoch": 蒸馏轮数}

        Returns:
            OrderDict: 模型的权重参数
        """
        self._init_kd(teacher, student, **kwargs)
        for _ in range(self.epoch):
            for _, (x, target) in enumerate(self.kd_loader):
                x, target = x.to(self.device), target.to(self.device)
                output = student(x)
                with torch.no_grad():
                    soft_target = teacher(x)
                # result = F.normalize(soft_target, dim=1, p=2)
                # soft_target = MSEloss(soft_target, output)
                loss = self.loss(
                    output,
                    soft_target=soft_target.detach(),
                    labels=target,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return student.state_dict()

    def multi_loss(self, method, soft_target_lst, output, target):
        """多个教师的知识蒸馏损失函数

        Args:
            method  :           str
                                蒸馏方法：
                                  1. avg_losses: 每个教师单独对学生进行蒸馏，最后加权求和
                                  2. avg_logits：每个教师的logits加权集成组成一个logits，再对学生进行蒸馏
                                  3. avg_probs：聚合每个教师的输出label，再对学生进行蒸馏

            output:
                                student的输出

            target:
                                真实标签值

        Returns:
            loss    :     蒸馏损失值
        """

        def f(soft_target=None, t_probs=None):
            return self.loss(
                output,
                soft_target=soft_target,
                labels=target,
                t_probs=t_probs,
            )

        # todo: AT_beta
        if method == "avg_losses":
            at_loss = 0
            """
            self.AT_beta = 0
            for soft_target in soft_target_lst:
                at_loss = at_loss + self.AT_beta * self.attention_diff(
                    output, soft_target
            )
            """
            student_loss_lst = [
                (f(soft_target.detach()) + at_loss) * weight
                for soft_target, weight in zip(soft_target_lst, self.weight_lst)
            ]
            return sum(student_loss_lst)
        elif method == "avg_logits":
            weight_soft_lst = [
                soft_target.detach() * weight
                for soft_target, weight in zip(soft_target_lst, self.weight_lst)
            ]
            return f(sum(weight_soft_lst))

        elif method == "avg_probs":
            teacher_probs_lst = [
                F.softmax(soft_target, dim=1) * weight
                for soft_target, weight in zip(soft_target_lst, self.weight_lst)
            ]
            return f(t_probs=sum(teacher_probs_lst))

        else:
            raise NotImplementedError("please input a vaild method")

    def multi(
        self, teacher_lst, student, method="avg_logits", weight_lst=None, **kwargs
    ):
        """多个教师对一个学生进行知识蒸馏

        Args:
            teacher_lst  :      list
                                教师模型组成的list

            student:
                                学生模型

            method:             str
                                蒸馏的方法, "avg_losses", "avg_logits", "avg_probs"。默认为"avg_logits",

            weight_lst:         list
                                每个teacher的权重

            kwargs:             dict
                                    {
                                        "lr":    学习率,
                                        "alpha": 蒸馏超参，蒸馏损失所占的权重
                                        "epoch": 蒸馏训练轮数
                                    }

        Returns:
            OrderDict: 模型的权重参数
        """
        self._init_kd(teacher_lst, student, **kwargs)
        if weight_lst == None:
            self.weight_lst = [1 / len(teacher_lst)] * len(teacher_lst)
        else:
            self.weight_lst = weight_lst

        for _ in range(self.epoch):
            for _, (x, target) in enumerate(self.kd_loader):
                x, target = x.to(self.device), target.to(self.device)
                output = self.student(x)

                soft_target_lst = []
                for teacher in self.teacher_lst:
                    with torch.no_grad():
                        soft_target_lst.append(teacher(x))

                loss = self.multi_loss(method, soft_target_lst, output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("train_loss_fine_tuning", loss.data)
        return student.state_dict()
