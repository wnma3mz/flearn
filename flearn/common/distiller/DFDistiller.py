# coding: utf-8
import torch

from .Distiller import Distiller


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
                output = self.student(x)

                soft_target_lst = []
                for teacher in self.teacher_lst:
                    with torch.no_grad():
                        soft_target = teacher(x)
                        soft_target_lst.append(soft_target)

                loss = self.multi_loss(method, soft_target_lst, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("train_loss_fine_tuning", loss.data)
        return student.state_dict()
