# coding: utf-8
import copy

from flearn.common.strategy import AVG


class Dyn(AVG):
    def __init__(self, h, alpha=0.01):
        super().__init__()
        self.h = h
        self.theta = copy.deepcopy(h)
        self.alpha = alpha

    def dyn_f(self, w_glob, w_local_lst):
        delta_theta = {}
        # assume agg_weight_lst all is 1.0
        for k in self.h.keys():
            delta_theta[k] = w_glob[k] * len(w_local_lst) - self.theta[k]

        for k in self.h.keys():
            # bn.num_batches_tracked 无法计算，Long与Float类型冲突
            if "weight" in k or "bias" in k:
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
