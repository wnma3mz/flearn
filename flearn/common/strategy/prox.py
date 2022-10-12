# coding: utf-8

import copy

from .avg import AVG


class Prox(AVG):
    """

    References
    ----------
    """

    def client_revice(self, trainer, server_p_bytes) -> None:
        super().client_revice(trainer, server_p_bytes)
        # 需要把服务器的模型复制给本地模型，方便计算loss
        trainer.server_model = copy.deepcopy(trainer.model)
        trainer.server_model.eval()
