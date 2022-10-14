# coding: utf-8
from typing import *

from .strategy import Strategy
from .utils import convert_to_np, convert_to_tensor

T = TypeVar("T")


class AVG(Strategy):
    """
    FederatedAveraging

    References
    ----------
    .. [1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial Intelligence and Statistics. PMLR, 2017: 1273-1282.
    """

    def client(self, trainer, agg_weight=1.0) -> Dict[str, T]:
        # step 1
        upload_p = {"agg_weight": agg_weight}
        upload_p["params"] = convert_to_np(trainer.weight)
        return upload_p

    def server(self, ensemble_params_lst, round_) -> Dict[str, T]:
        # step 2
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        try:
            w_glob = self.server_ensemble(agg_weight_lst, w_local_lst)
        except Exception as e:
            self.server_exception(e)

        return {"w_glob": w_glob}

    def client_receive(self, trainer, server_p_bytes) -> Dict[str, T]:
        # step 3
        # decode
        server_p = self.receive_processing(server_p_bytes)

        w_local = trainer.weight
        w_glob = convert_to_tensor(server_p["w_glob"])
        for k in w_glob.keys():
            w_local[k] = w_glob[k]

        trainer.model.load_state_dict(w_local)
        return server_p
