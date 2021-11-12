# coding: utf-8
import copy

import torch
import torch.nn.functional as F

from flearn.common.distiller import Distiller, KDLoss
from flearn.common.strategy import AVG


class DFDistiller(Distiller):
    def multi_loss(self, method, soft_target_lst, output):
        def f(soft_target):
            return self.loss(output, soft_target)

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
                for soft_target, weight in (soft_target_lst, self.weight_lst)
            ]
            return sum(student_loss_lst)
        elif method == "avg_logits":
            weight_soft_lst = [
                soft_target.detach() * weight
                for soft_target, weight in zip(soft_target_lst, self.weight_lst)
            ]
            return f(sum(weight_soft_lst))

        else:
            raise NotImplementedError("please input a vaild method")

    def multi(self, teacher_lst, student, method, weight_lst, **kwargs):
        self._init_kd(teacher_lst, student, **kwargs)
        self.weight_lst = weight_lst

        for _ in range(self.epoch):
            for _, (x, _) in enumerate(self.kd_loader):
                x = x.to(self.device)
                output = self.student(x)

                soft_target_lst = []
                for teacher in self.teacher_lst:
                    with torch.no_grad():
                        soft_target_lst.append(teacher(x))

                loss = self.multi_loss(method, soft_target_lst, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("train_loss_fine_tuning", loss.data)
        return student.state_dict()


class DF(AVG):
    """"""

    def __init__(self, model_fpath, model_base, device):
        super(DF, self).__init__(model_fpath)
        self.device = device
        self.model_base = model_base

    def server(self, ensemble_params_lst, round_, **kwargs):
        """
        kwargs:       dict
                        {
                            "lr":    学习率,
                            "T":     蒸馏超参，温度
                            "epoch": 蒸馏训练轮数
                            "method": 多个教师蒸馏一个学习的方法，avg_logits, avg_losses\
                            "kd_loader": 蒸馏数据集，仅需输入，无需标签
                        }
        """
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

        teacher_lst = []
        for w_local in w_local_lst:
            self.model_base.load_state_dict(w_local)
            teacher_lst.append(copy.deepcopy(self.model_base))

        self.model_base.load_state_dict(w_glob)
        student = copy.deepcopy(self.model_base)

        self.distiller = DFDistiller(
            kwargs["kd_loader"], self.device, kd_loss=KDLoss(kwargs["T"])
        )

        # agg_weight_lst：应该依照每个模型在验证集上的性能来进行分配
        w_glob = self.distiller.multi(
            teacher_lst, student, kwargs["method"], agg_weight_lst, **kwargs
        )

        return self.server_post_processing(w_glob, round_)
