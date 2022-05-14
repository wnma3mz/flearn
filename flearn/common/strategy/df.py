# coding: utf-8
import copy

import numpy as np
import torch

from flearn.common.distiller import Distiller, KDLoss

from .strategy import ParentStrategy


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

    def multi(
        self, teacher_lst, student, method="avg_logits", weight_lst=None, **kwargs
    ):
        self._init_kd(teacher_lst, student, **kwargs)
        if weight_lst == None:
            self.weight_lst = [1 / len(teacher_lst)] * len(teacher_lst)
        else:
            self.weight_lst = weight_lst

        for _ in range(self.epoch):
            for _, (x, _) in enumerate(self.kd_loader):
                x = x.to(self.device)
                _, _, output = self.student(x)

                soft_target_lst = []
                for teacher in self.teacher_lst:
                    with torch.no_grad():
                        _, _, soft_target = teacher(x)
                        soft_target_lst.append(soft_target)

                loss = self.multi_loss(method, soft_target_lst, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("train_loss_fine_tuning", loss.data)
        return student.state_dict()


class DF(ParentStrategy):
    """
    Ensemble distillation for robust model fusion in federated learning

    [1] Lin T, Kong L, Stich S U, et al. Ensemble distillation for robust model fusion in federated learning[J]. arXiv preprint arXiv:2006.07242, 2020.
    """

    def __init__(self, model_base, strategy):
        super().__init__(strategy)
        self.model_base = model_base

    def server_post_processing(self, ensemble_params_lst, ensemble_params, **kwargs):
        w_glob = ensemble_params["w_glob"]
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)

        teacher_lst = []
        for w_local in w_local_lst:
            self.model_base.load_state_dict(w_local)
            teacher_lst.append(copy.deepcopy(self.model_base))

        self.model_base.load_state_dict(w_glob)
        student = copy.deepcopy(self.model_base)

        kd_loader, device = kwargs.pop("kd_loader"), kwargs.pop("device")
        temperature = kwargs.pop("T")
        distiller = DFDistiller(
            kd_loader,
            device,
            kd_loss=KDLoss(temperature),
        )

        molecular = np.sum(agg_weight_lst)
        weight_lst = [w / molecular for w in agg_weight_lst]
        # agg_weight_lst：应该依照每个模型在验证集上的性能来进行分配
        ensemble_params["w_glob"] = distiller.multi(
            teacher_lst, student, kwargs.pop("method"), weight_lst=weight_lst, **kwargs
        )
        return ensemble_params

    def server(self, ensemble_params_lst, round_, **kwargs):
        """
        kwargs:       dict
                        {
                            "lr":    学习率,
                            "T":     蒸馏超参，温度
                            "epoch": 蒸馏训练轮数
                            "method": 多个教师蒸馏一个学习的方法，avg_logits, avg_losses
                            "kd_loader": 蒸馏数据集，仅需输入，无需标签
                        }
        """
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(
            ensemble_params_lst, ensemble_params, **kwargs
        )
