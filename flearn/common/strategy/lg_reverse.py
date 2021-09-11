# coding: utf-8
import pickle
from functools import reduce

from .strategy import Strategy


class LG_R(Strategy):
    """
    Fed Think locally, act globally, https://github.com/pliang279/LG-FedAvg

    R means that "not" in shared_key_layers

    References
    ----------
    .. [1] Liang P P, Liu T, Ziyin L, et al. Think locally, act globally: Federated learning with local and global representations[J]. arXiv preprint arXiv:2001.01523, 2020.
    """

    def __init__(self, model_fpath, shared_key_layers):
        super(LG_R, self).__init__(model_fpath)
        self.shared_key_layers = shared_key_layers

    def client(self, model_trainer, agg_weight=1.0):
        w_local = model_trainer.weight
        w_shared = {"params": {}, "agg_weight": agg_weight}
        for k in w_local.keys():
            if k not in self.shared_key_layers:
                w_shared["params"][k] = w_local[k].cpu()
        return w_shared

    def client_revice(self, model_trainer, w_glob_b):
        w_local = model_trainer.weight
        w_glob = pickle.loads(w_glob_b)
        for k in w_glob.keys():
            if k not in self.shared_key_layers:
                w_local[k] = w_glob[k]
        model_trainer.model.load_state_dict(w_local)
        return model_trainer.model

    def server(self, agg_weight_lst, w_local_lst, round_):
        try:
            key_lst = reduce(lambda x, y: set(x.keys()) & set(y.keys()), w_local_lst)
            key_lst = [k for k in key_lst if k not in self.shared_key_layers]

            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)
