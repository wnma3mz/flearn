# coding: utf-8
from .avg import AVG
from .utils import convert_to_tensor


class LG(AVG):
    """
    Fed Think locally, act globally, https://github.com/pliang279/LG-FedAvg

    References
    ----------
    .. [1] Liang P P, Liu T, Ziyin L, et al. Think locally, act globally: Federated learning with local and global representations[J]. arXiv preprint arXiv:2001.01523, 2020.
    """

    def __init__(self, shared_key_layers=None):
        super().__init__()
        self.shared_key_layers = shared_key_layers

    def client(self, trainer, agg_weight=1.0):
        w_shared = super().client(trainer, agg_weight)
        if self.shared_key_layers:
            all_key_lst = list(w_shared["params"].keys())
            delete_key_lst = [k for k in all_key_lst if k not in self.shared_key_layers]
            [w_shared["params"].pop(k) for k in delete_key_lst]
        return w_shared

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_glob = convert_to_tensor(data_glob_d["w_glob"])
        if self.shared_key_layers:
            key_lst = self.shared_key_layers
        else:
            key_lst = w_glob.keys()
        for k in key_lst:
            w_local[k] = w_glob[k]
        return w_local

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=self.shared_key_layers)
        except Exception as e:
            self.server_exception(e)
        if len(w_glob.keys()) < 10:
            print(w_glob.keys())
        return {"w_glob": w_glob}
