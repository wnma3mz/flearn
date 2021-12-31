# coding: utf-8

import copy

from flearn.common.strategy import AVG


class Distill(AVG):
    def client(self, trainer, agg_weight=1.0):
        w_shared = super(Distill, self).client(trainer, agg_weight)
        # upload logits
        w_shared["logits"] = trainer.logit_tracker.avg()
        return w_shared

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_glob, logits_glob = data_glob_d["w_glob"], data_glob_d["logits_glob"]
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
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

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

        delta_theta = {}
        # assume agg_weight_lst all is 1.0
        for k in self.h.keys():
            delta_theta[k] = w_glob[k] * len(w_local_lst) - self.theta[k]

        for k in self.h.keys():
            self.h[k] -= self.alpha / len(w_local_lst) * delta_theta[k]

        for k in self.h.keys():
            w_glob[k] = w_glob[k] - self.alpha * self.h[k]
        self.theta = w_glob

        return {"w_glob": w_glob}
