# coding: utf-8
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.common import Trainer
from flearn.common.strategy import AVG


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


class CCVRTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(CCVRTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.feat_lst = []
        self.label_lst = []

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
                self.feat_lst = []
                self.label_lst = []

        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def batch(self, data, target):
        h, _, output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            # 保存中间特征
            self.feat_lst.append(h)
            self.label_lst.append(target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class FedCCVR(AVG):
    """
    Federated learning via Classifier Calibration with Virtual Representations

    [1] Luo M, Chen F, Hu D, et al. No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data[J]. arXiv preprint arXiv:2106.05001, 2021.
    """

    def __init__(self, model_fpath, glob_model_base):
        super(FedCCVR, self).__init__(model_fpath)
        self.glob_model = glob_model_base

    def client(self, model_trainer, agg_weight=1.0):
        w_shared = super(FedCCVR, self).client(model_trainer, agg_weight)

        # 按照类别提取特征
        d = {}
        for h_l, label_l in zip(model_trainer.feat_lst, model_trainer.label_lst):
            for h, label in zip(h_l, label_l):
                label = int(label.cpu())
                if label not in d.keys():
                    d[label] = [h]
                else:
                    d[label].append(h)
        # 计算mu, sigma
        upload_d = {}
        for k, v in d.items():
            v_item = torch.stack(v).detach().cpu()
            mu, sigma = v_item.mean(dim=0), v_item.var(dim=0)
            upload_d[k] = {"mu": mu, "sigma": sigma, "N": len(v)}

        w_shared["fd"] = upload_d
        return w_shared

    @staticmethod
    def load_model(glob_model, glob_agg):
        glob_w = {}
        for k in glob_model.state_dict().keys():
            if k in glob_agg.keys():
                glob_w[k] = glob_agg[k]
        glob_model.load_state_dict(glob_w)
        return glob_model

    def server(self, ensemble_params_lst, round_, **kwargs):
        # 特征参数提取
        fd_lst = self.extract_lst(ensemble_params_lst, "fd")

        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

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

        # 重新训练分类器
        self.retrainer = ReTrain(fd_d, kwargs["device"])
        self.glob_model = self.load_model(self.glob_model, w_glob)
        w_train = self.retrainer.run(self.glob_model)

        for k in w_train.keys():
            w_glob[k] = w_train[k].cpu()

        return {"w_glob": w_glob}
