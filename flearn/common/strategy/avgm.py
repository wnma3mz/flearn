# coding: utf-8
import copy
import pickle

import numpy as np
import torch

from .avg import AVG


class AVGM(AVG):
    """
    Federate Learning Mean momentum

    References
    ----------
    .. [1] Hsu T M H, Qi H, Brown M. Measuring the effects of non-identical data distribution for federated visual classification[J]. arXiv preprint arXiv:1909.06335, 2019.
    """

    def client_revice(self, model_trainer, data_glob_b, beta=0.9):
        w_local = model_trainer.weight
        self.beta = beta

        w_glob = data_glob_b["w_glob"]
        delta_w = copy.deepcopy(w_glob)
        # 计算差值delta_w
        for k in w_glob.keys():
            delta_w[k] = delta_w[k] - w_local[k].cpu()

        # 初始化
        if "v_t" not in self.__dict__.keys():
            self.v_t = {k: torch.zeros_like(delta_w[k]) for k in delta_w.keys()}

        for k in w_glob.keys():
            self.v_t[k] = delta_w[k] + self.beta * self.v_t[k]

        for k in w_glob.keys():
            w_local[k] = w_local[k].cpu() + self.v_t[k]
        return w_local
