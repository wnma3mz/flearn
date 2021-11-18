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

    def client(self, model_trainer, agg_weight=1.0):
        w_shared = {"agg_weight": agg_weight}
        w_local = model_trainer.weight
        w_shared["params"] = {k: v.cpu() for k, v in w_local.items() if "bn" not in k}
        return w_shared

    def server(self, ensemble_params_lst, round_):
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            all_local_key_lst = [set(w_local.keys()) for w_local in w_local_lst]
            key_lst = reduce(lambda x, y: x & y, all_local_key_lst)
            key_lst = [k for k in key_lst if "bn" not in k]

            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        except Exception as e:
            return self.server_exception(e)
        return {"w_glob": w_glob}
