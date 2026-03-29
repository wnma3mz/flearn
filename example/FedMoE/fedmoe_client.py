# coding: utf-8
from flearn.client import Client


class FedMoEClient(Client):
    """Client wrapper that carries an expert id."""

    def __init__(self, conf, expert_id, **strategy_p):
        self.expert_id = expert_id
        super(FedMoEClient, self).__init__(conf, **strategy_p)
        # Avoid writing best checkpoints during algorithm development runs.
        self.save_best = False

    def train(self, i):
        res = super(FedMoEClient, self).train(i)
        res["expert_id"] = self.expert_id
        return res
