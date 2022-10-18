# coding: utf-8
import copy
from typing import *

from flearn.common.strategy import AVG

T = TypeVar("T")


class Distill(AVG):
    """
    Federated knowledge distillation

    [1] Seo H, Park J, Oh S, et al. Federated knowledge distillation[J]. arXiv preprint arXiv:2011.02367, 2020.
    """

    def client(self, trainer, agg_weight=1.0):
        w_shared = super(Distill, self).client(trainer, agg_weight)
        w_shared["logits"] = trainer.logits_tracker.avg()

        # In server do avg
        # w_shared["logits"] = trainer.logits_tracker
        # w_shared["labels"] = trainer.label_counts - 1
        return w_shared

    def server(self, ensemble_params_lst, round_):
        ensemble_params = super(Distill, self).server(ensemble_params_lst, round_)

        logits_lst = self.extract_lst(ensemble_params_lst, "logits")
        ensemble_params["glob_logits"] = self.aggregate_logits(logits_lst)

        # labels_lst = self.extract_lst(ensemble_params_lst, "labels")
        # ensemble_params["glob_logits"] = self.aggregate_logits_v2(logits_lst, labels_lst)

        return ensemble_params

    def client_receive(self, trainer, server_p_bytes) -> None:
        server_p = super(Distill, self).client_receive(trainer, server_p_bytes)
        trainer.glob_logits = copy.deepcopy(server_p["glob_logits"]).to(trainer.device)

    @staticmethod
    def aggregate_logits(logits_lst):
        user_logits = 0
        for item in logits_lst:
            user_logits += item
        return user_logits / len(logits_lst)

    @staticmethod
    def aggregate_logits_v2(logits_lst, labels_lst):
        user_logits, user_labels = logits_lst[0], labels_lst[0]
        for logits, labels in zip(logits_lst[1:], labels_lst[1:]):
            user_logits += logits
            user_labels += labels
        return user_logits / user_labels.float().unsqueeze(1)
