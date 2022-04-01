# coding: utf-8
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client.datasets.get_data import DictDataset, get_dataloader
from flearn.common import Trainer
from flearn.common.distiller import Distiller, KDLoss
from flearn.common.strategy import AVG, LG_R, ParentStrategy


class MD(LG_R):
    def __init__(self, shared_key_layers, glob_model, device):
        super(MD, self).__init__(shared_key_layers)
        self.glob_model = glob_model
        self.optimizer = optim.SGD(glob_model.parameters(), lr=1e-2)
        self.device = device
        self.glob_model.to(device)
        self.glob_model.train()

    @staticmethod
    def load_model(model_base_dict, w_dict):
        model_base_dict.update(w_dict)
        return model_base_dict

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_local_bak = copy.deepcopy(w_local)
        self.glob_model.load_state_dict(
            self.load_model(self.glob_model.state_dict(), w_local)
        )
        criterion = trainer.criterion
        x_lst, logits_lst = data_glob_d["x_lst"], data_glob_d["logits_lst"]

        # 为降低通信成本，该训练应该放到服务器端学习，再发回给各个客户端训练后不同的模型。但为了方便实现，该步骤先放到客户端进行
        epoch = 1
        for _ in range(epoch):
            for x, logits in zip(x_lst, logits_lst):
                x, logits = x.to(self.device), logits.to(self.device)
                y = self.glob_model(x)
                loss = criterion(y, logits)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        w_glob_model = self.glob_model.state_dict()
        for k in w_glob_model.keys():
            w_local_bak[k] = w_glob_model[k]

        return w_local_bak

    def client_pub_predict(self, w_local_lst, **kwargs):
        data_loader = kwargs["data_loader"]

        client_lst = []
        for w_local in w_local_lst:
            self.glob_model.load_state_dict(
                self.load_model(self.glob_model.state_dict(), w_local)
            )
            client_lst.append(copy.deepcopy(self.glob_model))

        x_lst = []
        logits_lst = []
        for x, _ in data_loader:
            x = x.to(self.device)

            logits = 0
            for client_m in client_lst:
                with torch.no_grad():
                    logits += client_m(x)

            logits /= len(w_local_lst)
            logits_lst.append(logits.cpu())
            x_lst.append(x.cpu())

        return x_lst, logits_lst

    def server(self, ensemble_params_lst, round_, **kwargs):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        x_lst, logits_lst = self.client_pub_predict(w_local_lst, **kwargs)
        return {"x_lst": x_lst, "logits_lst": logits_lst, "w_glob": ""}


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
        ensemble_params = super(Distill, self).server(ensemble_params_lst, round_)

        logits_lst = self.extract_lst(ensemble_params_lst, "logits")
        ensemble_params["logits_glob"] = self.aggregate_logits(logits_lst)
        return ensemble_params

    @staticmethod
    def aggregate_logits(logits_lst):
        user_logits = 0
        for item in logits_lst:
            user_logits += item
        return user_logits / len(logits_lst)


class Dyn(AVG):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.theta = copy.deepcopy(h)
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
        return w_glob

    def server_post_processing(self, ensemble_params_lst, ensemble_params):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        ensemble_params["w_glob"] = self.dyn_f(ensemble_params["w_glob"], w_local_lst)
        return ensemble_params

    def server(self, ensemble_params_lst, round_):
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(ensemble_params_lst, ensemble_params)


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


class CCVR(ParentStrategy):
    """
    Federated learning via Classifier Calibration with Virtual Representations

    [1] Luo M, Chen F, Hu D, et al. No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data[J]. arXiv preprint arXiv:2106.05001, 2021.
    """

    def __init__(self, head_model_base, strategy):
        super().__init__(strategy)
        self.head_model_base = head_model_base

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
            if len(v_item) < 10:
                continue
            # if len(v_item) * label_len * 2 < sum_:
            #     continue
            mu, sigma = v_item.mean(dim=0), v_item.var(dim=0)
            upload_d[k] = {"mu": mu, "sigma": sigma, "N": len(v)}
        return upload_d

    def client(self, trainer, agg_weight=1.0):
        w_shared = super().client(trainer, agg_weight)
        w_shared["fd"] = self.client_mean_feat(trainer.feat_lst, trainer.label_lst)
        return w_shared

    @staticmethod
    def load_model(model_base, new_model_dict):
        model_base_dict = model_base.state_dict()
        model_base_dict.update(new_model_dict)
        model_base.load_state_dict(model_base_dict)
        return model_base

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

    def server_post_processing(self, ensemble_params_lst, ensemble_params, **kwargs):
        # 特征参数提取
        fd_lst = self.extract_lst(ensemble_params_lst, "fd")
        fd_d = self.server_mean_feat(fd_lst)

        # 准备好数据集，模型、优化器等等
        trainset = DictDataset(fd_d)
        trainloader, _ = get_dataloader(trainset, trainset, batch_size=64)
        optimizer = optim.SGD(
            self.head_model_base.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.05
        )
        criterion = nn.CrossEntropyLoss()
        self.glob_model_base = self.load_model(
            self.head_model_base, ensemble_params["w_glob"]
        )

        # 重新训练分类器
        trainer = Trainer(
            self.head_model_base, optimizer, criterion, kwargs["device"], False
        )
        trainer.train(trainloader, epochs=1)
        w_train = trainer.weight

        for k in w_train.keys():
            ensemble_params["w_glob"][k] = w_train[k].cpu()

        return ensemble_params

    def server(self, ensemble_params_lst, round_, **kwargs):
        ensemble_params = super().server(ensemble_params_lst, round_)
        return self.server_post_processing(
            ensemble_params_lst, ensemble_params, **kwargs
        )


class DFCCVR(CCVR):
    def __init__(self, model_base, head_model_base, strategy):
        super().__init__(head_model_base, strategy)
        self.model_base = model_base
        self.df = DF(model_base, strategy)

    def server(self, ensemble_params_lst, round_, **kwargs):
        # 先DF后CCVR
        ensemble_params = self.df.server(ensemble_params_lst, round_, **kwargs)
        return self.server_post_processing(
            ensemble_params_lst, ensemble_params, **kwargs
        )
