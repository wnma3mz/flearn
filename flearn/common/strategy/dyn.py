# coding: utf-8
import copy
from typing import *

from flearn.common.strategy import AVG

T = TypeVar("T")


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

        not_k_lst = []
        for k in self.h.keys():
            # Float和Long类型不兼容
            try:
                self.h[k] -= self.alpha / len(w_local_lst) * delta_theta[k]
            except:
                not_k_lst.append(k)

        for k in self.h.keys():
            if k in not_k_lst:
                continue
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

    def client_receive(self, trainer, server_p_bytes) -> Dict[str, T]:
        super().client_receive(trainer, server_p_bytes)
        trainer.server_state_dict = copy.deepcopy(trainer.weight)
