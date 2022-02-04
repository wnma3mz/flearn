# coding: utf-8
import torch.nn as nn

from flearn.common import Trainer, init_strategy
from flearn.common.utils import setup_seed

setup_seed(0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(10 * 10, 2)

    def forward(self, x):
        x = x.view(-1, 10 * 10)
        x = self.fc(x)
        return x


class MyTrainer(Trainer):
    def __init__(self, model):
        self.model = model

    @property
    def weight(self):
        return self.model.state_dict()

    @property
    def grads(self):
        # not true, just test
        return self.model.state_dict()


if __name__ == "__main__":
    model_fpath = ""
    mytrainer = MyTrainer(MLP())
    s = init_strategy("avg", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    assert (
        upload_res["params"]["fc.weight"] == global_res["w_glob"]["fc.weight"]
    ).all()
    assert (upload_res["params"]["fc.bias"] == global_res["w_glob"]["fc.bias"]).all()

    assert (revice_res["fc.weight"] == global_res["w_glob"]["fc.weight"]).all()
    assert (revice_res["fc.bias"] == global_res["w_glob"]["fc.bias"]).all()

    s = init_strategy("opt", None)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    # todo
    s = init_strategy("avgm", None)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = init_strategy("bn", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)

    s = init_strategy("sgd", None)
    upload_res = s.client(mytrainer)
    revice_res = s.client_revice([upload_res], 0)

    s = init_strategy("lg", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = init_strategy("lg_r", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
    revice_res = s.client_revice(mytrainer, {"w_glob": s.client(mytrainer)["params"]})

    s = init_strategy("pav", None)
    upload_res = s.client(mytrainer)
    global_res = s.server([upload_res], 0)
