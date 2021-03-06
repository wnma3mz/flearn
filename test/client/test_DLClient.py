# coding: utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client import DLClient
from flearn.common.trainer import Trainer
from flearn.common.utils import setup_seed

setup_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(10 * 10, 2)

    def forward(self, x):
        x = x.view(-1, 10 * 10)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MLP()
    optim_ = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    input_ = torch.rand(size=(32, 10, 10))
    target = torch.randint(0, 2, size=(32,), dtype=torch.long)
    trainloader = ((input_, target), (input_, target))
    device = "cpu"
    dataset_name = "test"

    model_fpath = "."
    if not os.path.isdir(model_fpath):
        os.mkdir(model_fpath)

    client_id = 0
    c_conf = {
        "trainer": Trainer(model, optim_, nn.CrossEntropyLoss(), device, True),
        "trainloader": trainloader,
        "testloader": trainloader,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "dataset_name": dataset_name,
        "save": False,
        "display": True,
        "log": False,
    }

    c = DLClient(c_conf)
    c.run(0)
