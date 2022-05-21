# coding: utf-8
from functools import reduce

from .avg import AVG


class BN(AVG):
    """
    Federate Batch Normalization, https://github.com/med-air/FedBN

    References
    ----------
    .. [1] Li X, Jiang M, Zhang X, et al. FedBN: Federated Learning on Non-IID Features via Local Batch Normalization[J]. arXiv preprint arXiv:2102.07623, 2021.
    """

    def client(self, trainer, agg_weight=1.0):
        w_shared = super().client(trainer, agg_weight)
        all_key_lst = list(w_shared["params"].keys())
        delete_key_lst = [k for k in all_key_lst if "bn" in k]
        [w_shared["params"].pop(k) for k in delete_key_lst]
        return w_shared

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            all_local_key_lst = [set(w_local.keys()) for w_local in w_local_lst]
            key_lst = reduce(lambda x, y: x & y, all_local_key_lst)
            key_lst = [k for k in key_lst if "bn" not in k]

            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        except Exception as e:
            self.server_exception(e)
        return {"w_glob": w_glob}
