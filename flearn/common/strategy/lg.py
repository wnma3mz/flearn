# coding: utf-8
import pickle
from functools import reduce

from .strategy import Strategy


class LG(Strategy):
    """
    Fed Think locally, act globally, https://github.com/pliang279/LG-FedAvg

    References
    ----------
    .. [1] Liang P P, Liu T, Ziyin L, et al. Think locally, act globally: Federated learning with local and global representations[J]. arXiv preprint arXiv:2001.01523, 2020.
    """

    def __init__(self, model_fpath, shared_key_layers):
        super(LG, self).__init__(model_fpath)
        self.shared_key_layers = shared_key_layers

    def client(self, model_trainer, agg_weight=1.0):
        w_local = model_trainer.weight
        w_shared = {"params": {}, "agg_weight": agg_weight}
        for k in self.shared_key_layers:
            w_shared["params"][k] = w_local[k].cpu()
        return w_shared

    def client_revice(self, model_trainer, w_glob_b):
        w_local = model_trainer.weight
        w_glob = pickle.loads(w_glob_b)
        for k in self.shared_key_layers:
            w_local[k] = w_glob[k]
        model_trainer.model.load_state_dict(w_local)
        return model_trainer.model

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(
                agg_weight_lst, w_local_lst, key_lst=self.shared_key_layers
            )
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)
