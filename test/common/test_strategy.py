# coding: utf-8
import unittest

import numpy as np
import torch

from flearn.common.strategy.utils import convert_to_np, convert_to_tensor
from flearn.common.trainer import Trainer
from flearn.common.utils import (
    base_strategy_lst,
    setup_seed,
    setup_strategy,
    strategy_trainer_d,
)

setup_seed(0)


class Model:
    def __init__(self) -> None:
        self.d = {"fc.weight": np.ones((10, 10)), "fc.bias": np.zeros((10, 1))}

    def state_dict(self):
        return self.d

    def load_state_dict(self, w):
        for k, v in w.items():
            self.d[k] = v

    def to(self, x):
        pass

    def eval(self):
        pass


class MyTrainer(Trainer):
    def __init__(self):
        self.model = Model()

    @property
    def weight_o(self):
        return self.model.state_dict()

    @property
    def weight(self):
        return self.model.state_dict()

    @property
    def grads(self):
        # not true, just test
        return {"fc.weight": np.ones((10, 10)), "fc.bias": np.zeros((10, 1))}


class TestStrategy(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.trainer = MyTrainer()
        self.strategy = setup_strategy("avg", None)

    def test_client_receive(self):
        upload_res = self.strategy.client(self.trainer)
        global_res = self.strategy.server([upload_res], 0)
        self.strategy.client_receive(self.trainer, {"w_glob": self.strategy.client(self.trainer)["params"]})

        w_glob = convert_to_tensor(global_res["w_glob"])
        torch.testing.assert_allclose(self.trainer.weight["fc.weight"], w_glob["fc.weight"])
        torch.testing.assert_allclose(self.trainer.weight["fc.bias"], w_glob["fc.bias"])

    def test_all_strategy(self):
        for strategy in base_strategy_lst:
            if strategy in ["md", "pav"]:
                continue
            s = setup_strategy(strategy, None)
            upload_res = s.client(self.trainer)
            global_res = s.server([upload_res], 0)
            receive_res = s.client_receive(self.trainer, {"w_glob": s.client(self.trainer)["params"]})

        for strategy, trainer_o in strategy_trainer_d.items():
            if strategy in ["dyn", "moon"]:
                continue

            trainer = trainer_o(Model(), None, None, None)
            s = setup_strategy(strategy, None)
            upload_res = s.client(trainer)
            global_res = s.server([upload_res], 0)
            receive_res = s.client_receive(trainer, global_res)


if __name__ == "__main__":
    t = TestStrategy("test_client_receive")
    t.test_client_receive()
    t.test_all_strategy()
