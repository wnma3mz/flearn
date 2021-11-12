# coding: utf-8
import base64
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client import Client
from flearn.server import Server
from flearn.server.Communicator import Communicator as sc

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
    testloader = ((input_, target), (input_, target))
    device = "cpu"

    model_fpath = "."
    if not os.path.isdir(model_fpath):
        os.mkdir(model_fpath)
    dataset_name = "test"
    strategy_name = "avg"
    epoch = 1

    client_id = 0
    c_conf = {
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim_,
        "trainloader": trainloader,
        "testloader": testloader,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "device": device,
        "model_fpath": model_fpath,
        "epoch": epoch,
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "save": False,
        "display": True,
        "log": False,
    }

    c = Client(c_conf)
    client_lst = [c]
    i = 0
    train_res = c.train(i)
    upload_res = c.upload(i)

    s_conf = {
        "Round": 1,
        "client_numbers": 1,
        "model_fpath": ".",
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
    }

    s_model = Server(s_conf)
    data_lst = [upload_res]
    model_b64_str = s_model.ensemble(data_lst, 0)
    revice_res = c.revice(i, model_b64_str)

    evaluate_res = c.evaluate(i)
