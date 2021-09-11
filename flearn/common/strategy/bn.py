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
        w_local = model_trainer.weight
        w_shared = {"params": {}, "agg_weight": agg_weight}
        for k in w_local.keys():
            if "bn" not in k:
                w_shared["params"][k] = w_local[k].cpu()
        return w_shared

    def server(self, agg_weight_lst, w_local_lst, round_):
        try:
            key_lst = reduce(lambda x, y: set(x.keys()) & set(y.keys()), w_local_lst)
            key_lst = [k for k in key_lst if "bn" not in k]
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        except Exception as e:
            return self.server_exception(e)
        return self.server_post_processing(w_glob, round_)
