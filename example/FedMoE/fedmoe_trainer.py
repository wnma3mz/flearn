# coding: utf-8
from collections import OrderedDict

import torch
import torch.nn.functional as F

from flearn.common.trainer import Trainer


class FedMoETrainer(Trainer):
    """Trainer with backbone consistency regularization for FedMoE."""

    def __init__(self, model, optimizer, criterion, device, display=True, lambda_consistency=0.01):
        super(FedMoETrainer, self).__init__(model, optimizer, criterion, device, display)
        self.lambda_consistency = lambda_consistency
        self.backbone_prefixes = ("backbone", "backbone_proj")
        self.global_backbone_ref = None

    def set_global_backbone_ref(self, weights):
        if weights is None:
            self.global_backbone_ref = None
            return
        ref = OrderedDict()
        for k, v in weights.items():
            if self._is_backbone_key(k):
                if isinstance(v, torch.Tensor):
                    ref[k] = v.detach().clone().to(self.device)
                else:
                    ref[k] = torch.tensor(v, device=self.device)
        self.global_backbone_ref = ref

    def _is_backbone_key(self, key):
        return key.startswith(self.backbone_prefixes)

    def fed_loss(self):
        if self.global_backbone_ref is None:
            return torch.tensor(0.0, device=self.device)

        reg = torch.tensor(0.0, device=self.device)
        curr_state = self.model.state_dict()
        for key, value in curr_state.items():
            if key in self.global_backbone_ref and self._is_backbone_key(key):
                reg = reg + F.mse_loss(value, self.global_backbone_ref[key])
        return self.lambda_consistency * reg
