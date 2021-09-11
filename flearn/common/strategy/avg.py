# coding: utf-8
import copy
import pickle

from .strategy import Strategy


class AVG(Strategy):
    """
    FederatedAveraging

    References
    ----------
    .. [1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.
    """

    def client(self, model_trainer, agg_weight=1.0):
        w_local = model_trainer.weight
        w_shared = {"params": {}, "agg_weight": agg_weight}
        for k in w_local.keys():
            w_shared["params"][k] = w_local[k].cpu()
        return w_shared

    def client_revice(self, model_trainer, w_glob_b):
        w_local = model_trainer.weight
        w_glob = pickle.loads(w_glob_b)
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        model_trainer.model.load_state_dict(w_local)
        return model_trainer.model

    def server(self, agg_weight_lst, w_local_lst, round_):
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)
