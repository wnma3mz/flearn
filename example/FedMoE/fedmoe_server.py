# coding: utf-8
from typing import Dict

import numpy as np

from flearn.server import Server


class FedMoEServer(Server):
    """Server with a simple loss-aware router for expert(client) selection."""

    def __init__(self, conf, router_lr=0.2, router_temp=1.0, min_explore=0.15, **strategy_p):
        super(FedMoEServer, self).__init__(conf, **strategy_p)
        self.router_lr = router_lr
        self.router_temp = router_temp
        self.min_explore = min_explore
        self.router_logits: Dict[int, float] = {}
        self.ema_loss: Dict[int, float] = {}
        self.ema_momentum = 0.9

    def _init_client_state(self, lst):
        for cid in lst:
            if cid not in self.router_logits:
                self.router_logits[cid] = 0.0
                self.ema_loss[cid] = 1.0

    def _softmax_sample(self, client_ids, k):
        logits = np.array([self.router_logits[cid] for cid in client_ids], dtype=np.float64)
        logits = logits / max(self.router_temp, 1e-6)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)

        # Mix exploration to avoid routing collapse in early rounds.
        p = (1.0 - self.min_explore) * p + self.min_explore / len(client_ids)
        p = p / np.sum(p)
        return np.random.choice(client_ids, size=k, replace=False, p=p)

    def active_client(self, lst, k):
        if k == -1:
            self._init_client_state(lst)
            return lst

        k = min(len(lst), k)
        if k <= 0:
            raise SystemError("Please input vaild k")

        self._init_client_state(lst)
        selected = self._softmax_sample(list(lst), k)
        return selected.tolist()

    def train(self, data_lst):
        losses, train_accs = [], []
        for item in data_lst:
            cid = item["client_id"]
            loss = float(item["loss"])
            acc = float(item["train_acc"])
            losses.append(loss)
            train_accs.append(acc)

            self.ema_loss[cid] = self.ema_momentum * self.ema_loss.get(cid, loss) + (1 - self.ema_momentum) * loss

        baseline = float(np.mean(losses))
        for item in data_lst:
            cid = item["client_id"]
            advantage = float(item["loss"]) - baseline
            # Minimize expected loss: decrease logits for high-loss clients.
            self.router_logits[cid] = self.router_logits.get(cid, 0.0) - self.router_lr * advantage

        idx_lst = [x["client_id"] for x in data_lst]
        return float(np.mean(losses)), float(np.mean(train_accs)), idx_lst
