# coding: utf-8
from flearn.common.strategy import AVG


class Distill(AVG):
    """
    Federated knowledge distillation

    [1] Seo H, Park J, Oh S, et al. Federated knowledge distillation[J]. arXiv preprint arXiv:2011.02367, 2020.
    """

    def client(self, trainer, agg_weight=1.0):
        w_shared = super(Distill, self).client(trainer, agg_weight)
        # upload logits
        w_shared["logits"] = trainer.logit_tracker.avg()
        return w_shared

    def server(self, ensemble_params_lst, round_):
        ensemble_params = super(Distill, self).server(ensemble_params_lst, round_)

        logits_lst = self.extract_lst(ensemble_params_lst, "logits")
        ensemble_params["logits_glob"] = self.aggregate_logits(logits_lst)
        return ensemble_params

    def client_revice(self, trainer, data_glob_d):
        w_local = super(Distill, self).client_revice(trainer, data_glob_d)
        logits_glob = data_glob_d["logits_glob"]
        return w_local, logits_glob

    @staticmethod
    def aggregate_logits(logits_lst):
        user_logits = 0
        for item in logits_lst:
            user_logits += item
        return user_logits / len(logits_lst)
