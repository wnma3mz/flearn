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

    def client(self, model_trainer, agg_weight=1.0):
        g_local = model_trainer.grads
        g_shared = {"params": {}, "agg_weight": agg_weight}
        for k in g_local.keys():
            g_shared["params"][k] = g_local[k].cpu()
        return g_shared

    def client_revice(self, model_trainer, w_glob_b):
        g_glob = pickle.loads(w_glob_b)
        w_local = model_trainer.weight
        for k, v in w_local.items():
            w_local[k] = v.cpu() + g_glob[k]
        model_trainer.model.load_state_dict(w_local)
        return model_trainer.model
