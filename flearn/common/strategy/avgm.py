# coding: utf-8
import copy

import numpy as np

from .avg import AVG
from .utils import convert_to_np, convert_to_tensor


class AVGM(AVG):
    """
    Federate Learning Mean momentum

    References
    ----------
    .. [1] Hsu T M H, Qi H, Brown M. Measuring the effects of non-identical data distribution for federated visual classification[J]. arXiv preprint arXiv:1909.06335, 2019.
    """

    def mean_momentum(self, w_local, w_glob, beta):
        self.beta = beta

        delta_w = copy.deepcopy(w_glob)
        # 计算差值delta_w
        for k in w_glob.keys():
            delta_w[k] = delta_w[k] - w_local[k]

        # 初始化
        if "v_t" not in self.__dict__.keys():
            self.v_t = {k: np.zeros_like(delta_w[k]) for k in delta_w.keys()}

        for k in w_glob.keys():
            self.v_t[k] = delta_w[k] + self.beta * self.v_t[k]

        for k in w_glob.keys():
            w_local[k] = w_local[k] + self.v_t[k]
        return w_local

    def client_revice(self, trainer, server_p_bytes, beta=0.9):
        server_p = self.revice_processing(server_p_bytes)

        w_local = convert_to_np(trainer.weight)
        w_local = self.mean_momentum(w_local, server_p["w_glob"], beta)

        trainer.model.load_state_dict(convert_to_tensor(w_local))
        return server_p
