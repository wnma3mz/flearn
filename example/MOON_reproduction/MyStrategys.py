# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client.datasets.get_data import DictDataset, get_dataloader
from flearn.common import Trainer
from flearn.common.strategy import DF, ParentStrategy


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
