# coding: utf-8
import copy

from .avg import AVG
from .utils import convert_to_np, convert_to_tensor


class SGD(AVG):
    """
    Federated SGD

    References
    ----------
    .. [1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.

    """

    def client(self, trainer, agg_weight=1.0):
        g_shared = {"agg_weight": agg_weight}
        g_shared["params"] = convert_to_np(trainer.grads)
        return g_shared

    def client_revice(self, trainer, server_p_bytes):
        server_p = self.revice_processing(server_p_bytes)

        g_glob = convert_to_tensor(server_p["w_glob"])

        w_local = copy.deepcopy(trainer.weight_o)
        for k, v in w_local.items():
            w_local[k] = v.cpu() + g_glob[k]

        trainer.model.load_state_dict(convert_to_tensor(w_local))
        return server_p
