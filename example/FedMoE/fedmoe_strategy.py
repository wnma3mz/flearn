# coding: utf-8
from typing import Dict, TypeVar

from flearn.common.strategy import Strategy
from flearn.common.strategy.utils import convert_to_np, convert_to_tensor

T = TypeVar("T")


class FedMoEStrategy(Strategy):
    """Aggregate only backbone; keep expert layers local."""

    def __init__(self, backbone_prefixes=("backbone", "backbone_proj")):
        super(FedMoEStrategy, self).__init__()
        self.backbone_prefixes = backbone_prefixes

    def _is_backbone_key(self, key):
        return key.startswith(self.backbone_prefixes)

    def client(self, trainer, agg_weight=1.0) -> Dict[str, T]:
        upload_p = {"agg_weight": agg_weight}
        upload_p["params"] = convert_to_np(trainer.weight)
        return upload_p

    def server(self, ensemble_params_lst, round_) -> Dict[str, T]:
        agg_weight_lst, w_local_lst = self.server_pre_processing(ensemble_params_lst)
        key_lst = [k for k in w_local_lst[0].keys() if self._is_backbone_key(k)]
        w_glob = self.server_ensemble(agg_weight_lst, w_local_lst, key_lst=key_lst)
        return {"w_glob": w_glob}

    def client_receive(self, trainer, server_p_bytes) -> Dict[str, T]:
        server_p = self.receive_processing(server_p_bytes)
        w_local = trainer.weight
        w_glob = convert_to_tensor(server_p["w_glob"])
        for key in w_glob.keys():
            if self._is_backbone_key(key):
                w_local[key] = w_glob[key]
        trainer.model.load_state_dict(w_local)
        # Keep a snapshot for consistency regularization in the next local step.
        if hasattr(trainer, "set_global_backbone_ref"):
            trainer.set_global_backbone_ref(w_glob)
        return server_p
