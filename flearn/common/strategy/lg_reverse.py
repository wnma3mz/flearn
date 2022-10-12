# coding: utf-8
from .avg import AVG
from .utils import convert_to_tensor


class LG_R(AVG):
    """
    Fed Think locally, act globally, https://github.com/pliang279/LG-FedAvg

    R means that "not" in shared_key_layers

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
            delete_key_lst = [k for k in all_key_lst if k in self.shared_key_layers]
            [w_shared["params"].pop(k) for k in delete_key_lst]
        return w_shared

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            self.server_exception(e)
        if len(w_glob.keys()) < 10:
            print(w_glob.keys())
        return {"w_glob": w_glob}

    def client_revice(self, trainer, server_p_bytes):
        server_p = self.revice_processing(server_p_bytes)

        w_local = trainer.weight
        w_glob = convert_to_tensor(server_p["w_glob"])

        key_lst = self.shared_key_layers if self.shared_key_layers else []
        for k in w_glob.keys():
            if k not in key_lst:
                w_local[k] = w_glob[k]
        trainer.model.load_state_dict(w_local)
        return server_p
