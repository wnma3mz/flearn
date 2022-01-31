# coding: utf-8
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.common.distiller import Distiller, KDLoss
from flearn.common.strategy import AVG


class Distill(AVG):
    def client(self, trainer, agg_weight=1.0):
        w_shared = super(Distill, self).client(trainer, agg_weight)
        # upload logits
        w_shared["logits"] = trainer.logit_tracker.avg()
        return w_shared

    def client_revice(self, trainer, data_glob_d):
        w_local = super(Distill, self).client_revice(trainer, data_glob_d)
        logits_glob = data_glob_d["logits_glob"]
        return w_local, logits_glob

    def server(self, ensemble_params_lst, round_):
        g_shared = super(Distill, self).server(ensemble_params_lst, round_)

        logits_lst = self.extract_lst(ensemble_params_lst, "logits")
        g_shared["logits_glob"] = self.aggregate_logits(logits_lst)
        return g_shared

    def aggregate_logits(self, logits_lst):
        user_logits = 0
        for item in logits_lst:
            user_logits += item
        return user_logits / len(logits_lst)


class Dyn(AVG):
    def __init__(self, model_fpath, h):
        super(Dyn, self).__init__(model_fpath)
        self.h = h
        self.theta = copy.deepcopy(self.h)
        self.alpha = 0.01

    def dyn_f(self, w_glob, w_local_lst):
        delta_theta = {}
        # assume agg_weight_lst all is 1.0
        for k in self.h.keys():
            delta_theta[k] = w_glob[k] * len(w_local_lst) - self.theta[k]

        for k in self.h.keys():
            self.h[k] -= self.alpha / len(w_local_lst) * delta_theta[k]

        for k in self.h.keys():
            w_glob[k] = w_glob[k] - self.alpha * self.h[k]
        self.theta = w_glob

    def server(self, ensemble_params_lst, round_):
        g_shared = super(Dyn, self).server(ensemble_params_lst, round_)
        _, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        self.dyn_f(g_shared["w_glob"], w_local_lst)
        return g_shared


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
                        soft_target_lst.append(teacher(x))

                loss = self.multi_loss(method, soft_target_lst, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("train_loss_fine_tuning", loss.data)
        return student.state_dict()


class DF(AVG):
    """
    Ensemble distillation for robust model fusion in federated learning

    [1] Lin T, Kong L, Stich S U, et al. Ensemble distillation for robust model fusion in federated learning[J]. arXiv preprint arXiv:2006.07242, 2020.
    """

    def __init__(self, model_fpath, model_base):
        super(DF, self).__init__(model_fpath)
        self.model_base = model_base

    def ensemble_w(self, ensemble_params_lst, w_glob, **kwargs):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)

        teacher_lst = []
        for w_local in w_local_lst:
            self.model_base.load_state_dict(w_local)
            teacher_lst.append(copy.deepcopy(self.model_base))

        self.model_base.load_state_dict(w_glob)
        student = copy.deepcopy(self.model_base)

        self.distiller = DFDistiller(
            kwargs.pop("kd_loader"),
            kwargs.pop("device"),
            kd_loss=KDLoss(kwargs.pop("T")),
        )

        molecular = np.sum(agg_weight_lst)
        weight_lst = [w / molecular for w in agg_weight_lst]
        # agg_weight_lst：应该依照每个模型在验证集上的性能来进行分配
        w_glob = self.distiller.multi(
            teacher_lst, student, kwargs.pop("method"), weight_lst=weight_lst, **kwargs
        )
        return w_glob

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
        g_shared = super(DF, self).server(ensemble_params_lst, round_)
        g_shared["w_glob"] = self.ensemble_w(
            ensemble_params_lst, g_shared["w_glob"], **kwargs
        )
        return g_shared


class ReTrain:
    def __init__(self, fd_d, device):
        self.fd_d = fd_d
        self.device = device

    def run(self, student, lr=0.01):
        student.train()
        student.to(self.device)
        CELoss = nn.CrossEntropyLoss().to(self.device)

        print(lr)
        optimizer = optim.SGD(
            student.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9
        )
        for _ in range(1):
            loss_lst = []
            for target, x in self.fd_d.items():
                target = torch.tensor([target] * x.size()[0])
                x = x.type(torch.float32)
                x, target = x.to(self.device), target.to(self.device)

                output = student(x)

                loss = CELoss(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_lst.append(loss.data.item())
            print("Loss: SUM {:.4f}".format(np.mean(loss_lst)))
        return student.state_dict()


class CCVR(Distill, Dyn, DF):
    def __init__(
        self,
        model_fpath,
        glob_model_base,
        strategy=None,
        h=None,
        shared_key_layers=None,
    ):
        super(CCVR, self).__init__(model_fpath, h)
        self.glob_model = glob_model_base
        self.strategy = strategy
        self.shared_key_layers = shared_key_layers

    @staticmethod
    def client_mean_feat(feat_lst, label_lst):
        sum_ = 0
        # 按照类别提取特征
        d = {}
        for h_l, label_l in zip(feat_lst, label_lst):
            sum_ += len(h_l)
            for h, label in zip(h_l, label_l):
                label = int(label.cpu())
                if label not in d.keys():
                    d[label] = [h]
                else:
                    d[label].append(h)
        # label_len = len(d.keys())

        # 计算mu, sigma
        upload_d = {}
        for k, v in d.items():
            v_item = torch.stack(v).detach().cpu()
            # 考虑样本数量过少不上传的情况
            # if len(v_item) * label_len * 2 < sum_:
            #     continue
            mu, sigma = v_item.mean(dim=0), v_item.var(dim=0)
            upload_d[k] = {"mu": mu, "sigma": sigma, "N": len(v)}
        return upload_d

    def client(self, trainer, agg_weight=1.0):
        if self.strategy == "lg":
            w_shared = {"agg_weight": agg_weight}
            w_local = trainer.weight
            w_shared["params"] = {k: w_local[k].cpu() for k in self.shared_key_layers}
        else:
            w_shared = super(CCVR, self).client(trainer, agg_weight)

        if self.strategy == "distill":
            w_shared["logits"] = trainer.logit_tracker.avg()

        w_shared["fd"] = self.client_mean_feat(trainer.feat_lst, trainer.label_lst)
        return w_shared

    @staticmethod
    def load_model(glob_model, glob_agg):
        glob_w = {}
        for k in glob_model.state_dict().keys():
            if k in glob_agg.keys():
                glob_w[k] = glob_agg[k]
        glob_model.load_state_dict(glob_w)
        return glob_model

    def server_mean_feat(self, fd_lst):
        # 获取每个客户端的标签，集成一起
        label_lst = []
        for x in fd_lst:
            label_lst += list(x.keys())
        label_lst = list(set(label_lst))
        print("labels: ", label_lst)

        fd_d = {}
        # 统计每个标签的特征分布
        for l in label_lst:
            # 客户端不一定具备所有的标签，异质
            labeled_fd_lst = [x for x in fd_lst if l in x.keys()]
            sum_n = sum(x[l]["N"] for x in labeled_fd_lst)

            mu_lst = [fd[l]["mu"] * fd[l]["N"] / sum_n for fd in labeled_fd_lst]
            mu = torch.stack(mu_lst).sum(dim=0)

            sigma1 = torch.stack(
                [fd[l]["mu"] * (fd[l]["N"] - 1) / (sum_n - 1) for fd in labeled_fd_lst]
            ).sum(dim=0)
            sigma2 = torch.stack(
                [
                    fd[l]["mu"] * fd[l]["mu"].T * fd[l]["N"] / (sum_n - 1)
                    for fd in labeled_fd_lst
                ]
            ).sum(dim=0)

            sigma = sigma1 + sigma2 - sum_n / (sum_n - 1) * mu * mu.T

            # 生成batchsize为200的数据样本，总共有10类，所以就是2k个样本
            dist_c = np.random.normal(mu, sigma, size=(200, mu.size()[0]))
            fd_d[l] = torch.tensor(dist_c)

        return fd_d

    def server(self, ensemble_params_lst, round_, **kwargs):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)
        g_shared = {"w_glob": w_glob}

        if self.strategy == "dyn":
            _, w_local_lst = self.server_pre_processing(ensemble_params_lst)
            self.dyn_f(g_shared["w_glob"], w_local_lst)
        elif self.strategy == "distill":
            logits_lst = self.extract_lst(ensemble_params_lst, "logits")
            g_shared["logits_glob"] = self.aggregate_logits(logits_lst)
        elif self.strategy == "df":
            g_shared["w_glob"] = self.ensemble_w(
                ensemble_params_lst, g_shared["w_glob"], **kwargs
            )

        # 特征参数提取
        fd_lst = self.extract_lst(ensemble_params_lst, "fd")
        fd_d = self.server_mean_feat(fd_lst)

        # 重新训练分类器
        self.retrainer = ReTrain(fd_d, kwargs["device"])
        self.glob_model = self.load_model(self.glob_model, g_shared["w_glob"])
        w_train = self.retrainer.run(self.glob_model)

        for k in w_train.keys():
            g_shared["w_glob"][k] = w_train[k].cpu()

        return g_shared

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_glob = data_glob_d["w_glob"]
        for k in w_glob.keys():
            w_local[k] = w_glob[k]

        if self.strategy == "distill":
            logits_glob = data_glob_d["logits_glob"]
            return w_local, logits_glob

        return w_local
