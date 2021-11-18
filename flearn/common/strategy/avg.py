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
        w_shared = {"agg_weight": agg_weight}
        w_local = model_trainer.weight
        w_shared["params"] = {k: v.cpu() for k, v in w_local.items()}
        return w_shared

    def client_revice(self, model_trainer, data_glob_b):
        w_local = model_trainer.weight
        w_glob = data_glob_b["w_glob"]
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        return w_local

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            return self.server_exception(e)

        return {"w_glob": w_glob}
