# coding: utf-8
import copy
import pickle

from .avg import AVG


class SGD(AVG):
    """
    Federated SGD

    References
    ----------
    .. [1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.

    """

    def client(self, trainer, agg_weight=1.0):
        g_shared = {"agg_weight": agg_weight}
        g_local = trainer.grads
        g_shared["params"] = {k: v.cpu() for k, v in g_local.items()}
        return g_shared

    def client_revice(self, trainer, data_glob_d):
        g_glob = data_glob_d["w_glob"]

        w_local = trainer.weight
        for k, v in w_local.items():
            w_local[k] = v.cpu() + g_glob[k]
        return w_local
