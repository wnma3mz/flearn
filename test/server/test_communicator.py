# coding: utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client import Client
from flearn.common.trainer import Trainer
from flearn.common.utils import setup_seed
from flearn.server import Server
from flearn.server.Communicator import Communicator as sc

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


def test_sc(conf):
    server_o = sc(conf=conf)
    server_o.max_workers = 1
    for ri in range(conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=1)
    return loss, train_acc, test_acc


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
        "trainer": Trainer(model, optim_, nn.CrossEntropyLoss(), device, True),
        "trainloader": trainloader,
        "testloader": testloader,
        # "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": epoch,
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "save": False,
        "log": False,
    }

    client_lst = [Client(c_conf)]

    # 常规情况
    s_conf = {
        "model_fpath": ".",
        "strategy_name": strategy_name,
    }
    sc_conf = {
        "server": Server(s_conf),
        "client_numbers": len(client_lst),
        "Round": 1,
        "dataset_name": dataset_name,
        "client_lst": client_lst,
    }
    test_sc(conf=sc_conf)

    # 服务器端进行测试, 但客户端不进行测试
    s_conf["eval_conf"] = {
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "device": device,
        "display": False,
        "dataloader": testloader,
        "eval_clients": False,
        "trainer": Trainer,
    }
    sc_conf["server"] = Server(s_conf)
    test_sc(conf=sc_conf)

    # 服务器端进行测试, 客户端也进行测试
    s_conf["eval_conf"]["eval_clients"] = True
    sc_conf["server"] = Server(s_conf)
    test_sc(conf=sc_conf)
