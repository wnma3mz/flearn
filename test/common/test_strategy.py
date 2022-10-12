# coding: utf-8
import numpy as np

from flearn.common.strategy.utils import convert_to_np, convert_to_tensor
from flearn.common.trainer import Trainer
from flearn.common.utils import setup_seed, setup_strategy

setup_seed(0)


class Model:
    def __init__(self) -> None:
        self.d = {"fc.weight": np.ones((10, 10)), "fc.bias": np.zeros((10, 1))}

    def state_dict(self):
        return self.d

    def load_state_dict(self, w):
        for k, v in w.items():
            self.d[k] = v


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


if __name__ == "__main__":
    model_fpath = ""
    mytrainer = MyTrainer()
    s = setup_strategy("avg", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    w_glob = convert_to_tensor(global_res["w_glob"])
    assert (mytrainer.weight["fc.weight"] == w_glob["fc.weight"]).all()
    assert (mytrainer.weight["fc.bias"] == w_glob["fc.bias"]).all()

    s = setup_strategy("opt", None)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    # todo
    s = setup_strategy("avgm", None)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = setup_strategy("bn", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)

    # s = setup_strategy("sgd", None)
    # upload_res = s.client(mytrainer)
    # global_res = s.server([upload_res], 0)
    # revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = setup_strategy("lg", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = setup_strategy("lg_r", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    # s = setup_strategy("pav", None)
    # upload_res = s.client(mytrainer)
    # global_res = s.server([upload_res], 0)
