# coding: utf-8
import copy
import pickle

from .strategy import Strategy


class SGD(Strategy):
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

    def server(self, agg_weight_lst, w_local_lst, round_):
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)
