# coding: utf-8
import copy

import numpy as np

from .avg import AVG
from .utils import convert_to_np, convert_to_tensor


class OPT(AVG):
    """
    Fed Optimization, https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadagrad.py

    1. Adagrad
    2. Yogi
    3. Adam

    References
    ----------
    .. [1] Reddi S, Charles Z, Zaheer M, et al. Adaptive federated optimization[J]. arXiv preprint arXiv:2003.00295, 2020.
    """

    def adaptive_opt(self, w_local, w_glob, method):
        self.eta = 1e-1
        self.tau = 1e-9
        self.beta1 = 0.9
        self.beta2 = 0.99

        delta_w = copy.deepcopy(w_glob)
        # 计算差值delta_w
        for k in w_glob.keys():
            delta_w[k] = w_glob[k] - w_local[k]

        # 原始delta_t
        # 初始化delta_t
        # if "delta_t" not in self.__dict__.keys():
        #     self.delta_t = {k: np.zeros_like(delta_w[k]) for k in delta_w.keys()}
        # for k in w_glob.keys():
        #     self.delta_t[k] = (
        #         self.beta1 * delta_w[k] + (1 - self.beta1) * self.delta_t[k]
        #     )
        # 简化版delta_t
        self.delta_t = delta_w
        # 初始化v_t
        if "v_t" not in self.__dict__.keys():
            self.v_t = {k: np.zeros_like(self.delta_t[k]) for k in self.delta_t.keys()}

        # a) Adagrad (Best)
        # b) Yogi
        # c) Adam
        multi_delta_t = {k: np.multiply(self.delta_t[k], self.delta_t[k]) for k in self.delta_t.keys()}
        if method == "adagrad":
            self.v_t = {k: self.v_t[k] + multi_delta_t[k] for k in self.delta_t.keys()}
        elif method == "yogi":
            self.v_t = {
                k: self.v_t[k] - (1 - self.beta2) * multi_delta_t[k] * np.sign(self.v_t[k] - multi_delta_t[k])
                for k in self.delta_t.keys()
            }
        elif method == "adam":
            self.v_t = {k: self.beta2 * self.v_t[k] + (1 - self.beta2) * multi_delta_t[k] for k in self.delta_t.keys()}

        for k in w_glob.keys():
            w_local[k] = w_local[k] + self.eta * self.delta_t[k] / (np.sqrt(self.v_t[k]) + self.tau)

        return w_local

    def client_revice(self, trainer, server_p_bytes, method="Adagrad"):
        server_p = self.revice_processing(server_p_bytes)

        method = method.lower()
        assert method in ["adagrad", "yogi", "adam"]
        w_local = convert_to_np(trainer.weight)
        w_local = self.adaptive_opt(w_local, server_p["w_glob"], method)

        trainer.model.load_state_dict(convert_to_tensor(w_local))
        return server_p
