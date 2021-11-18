# coding: utf-8
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.distiller import KDLoss
from flearn.common.strategy import AVG


class LogitTracker:
    """
    https://github.com/zhuangdizhu/FedGen/blob/HEAD/FLAlgorithms/users/userFedDistill.py
    """

    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels)  # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels, unique_labels))

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


class Distill(AVG):
    """
    Federated knowledge distillation

    [1] Seo H, Park J, Oh S, et al. Federated knowledge distillation[J]. arXiv preprint arXiv:2011.02367, 2020.
    """

    def client(self, model_trainer, agg_weight=1.0):
        w_shared = super(Distill, self).client(model_trainer, agg_weight)
        # upload logits
        w_shared["logits"] = model_trainer.logit_tracker.avg()
        return w_shared

    def client_revice(self, model_trainer, data_glob_d):
        w_local = model_trainer.weight
        w_glob, logits_glob = data_glob_d["w_glob"], data_glob_d["logits_glob"]
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        return w_local, logits_glob

    def server(self, ensemble_params_lst, round_):
        g_shared = super(Distill, self).client(ensemble_params_lst, round_)

        logits_lst = self.extract_lst(ensemble_params_lst, "logits")
        g_shared["logits_glob"] = self.aggregate_logits(logits_lst)

        return g_shared

    def aggregate_logits(self, logits_lst):
        user_logits = 0
        for item in logits_lst:
            user_logits += item
        return user_logits / len(logits_lst)


class DistillClient(Client):
    def revice(self, i, glob_params):
        # decode
        w_glob_b = self.encrypt.decode(glob_params)
        # update
        update_w, logits_glob = self.strategy.client_revice(
            self.model_trainer, w_glob_b
        )

        if self.scheduler != None:
            self.scheduler.step()
        self.model_trainer.model.load_state_dict(update_w)
        self.model_trainer.glob_logit = copy.deepcopy(logits_glob)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class DistillTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super(DistillTrainer, self).__init__(
            model, optimizer, criterion, device, display
        )
        self.logit_tracker = LogitTracker(10)  # cifar10
        self.glob_logit = None
        self.kd_mu = 1
        self.kd_loss = KDLoss(2)

    def batch(self, data, target):
        output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            if self.glob_logit != None:
                self.glob_logit = self.glob_logit.to(self.device)
                target_p = self.glob_logit[target, :]
                loss += self.kd_mu * self.kd_loss(output, target_p)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新上传的logits
            self.logit_tracker.update(output, target)

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc
