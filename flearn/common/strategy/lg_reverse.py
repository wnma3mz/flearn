# coding: utf-8
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

    def client(self, trainer, agg_weight=1.0):
        w_shared = {"agg_weight": agg_weight}
        w_local = trainer.weight
        w_shared["params"] = {
            k: v.cpu() for k, v in w_local.items() if k not in self.shared_key_layers
        }
        return w_shared

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_glob = data_glob_d["w_glob"]
        for k in w_glob.keys():
            if k not in self.shared_key_layers:
                w_local[k] = w_glob[k]
        return w_local

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            self.server_exception(e)
        return {"w_glob": w_glob}
