# coding: utf-8
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from FedCVVR import CCVRTrainer, FedCCVR
from flearn.client import Client
from flearn.client.utils import get_free_gpu_id
from flearn.common.utils import setup_seed
from flearn.server import Communicator as sc
from model import GlobModel, ModelFedCon
from MyClients import MOONClient, ProxClient
from utils import get_dataloader, partition_data

# 设置随机数种子
setup_seed(0)

idx = get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")

parser = argparse.ArgumentParser(description="Please input strategy_name")
parser.add_argument(
    "--strategy_name", dest="strategy_name", choices=["avg", "moon", "prox"]
)
parser.add_argument("--local_epoch", dest="local_epoch", default=1, type=int)
parser.add_argument("--frac", dest="frac", default=1, type=float)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument("--iid", dest="iid", action="store_true")
parser.add_argument(
    "--dataset_name",
    dest="dataset_name",
    default="mnist",
    choices=["tinyimagenet", "cifar10", "cifar100"],
    type=str,
)
parser.add_argument(
    "--dataset_fpath",
    dest="dataset_fpath",
    type=str,
)

args = parser.parse_args()
iid = args.iid
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_strategy = args.strategy_name.lower()
# 设置数据集
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath
client_numbers = 10

batch_size = 64
if iid == True:
    partition = "homo"
else:
    partition = "noniid"
    beta = 0.5
print("切分{}数据集, 切割方式: {}".format(dataset_name, partition))
(
    X_train,
    y_train,
    X_test,
    y_test,
    net_dataidx_map,
    traindata_cls_counts,
) = partition_data(
    dataset_name,
    dataset_fpath,
    logdir="./logs",
    partition=partition,
    n_parties=client_numbers,
    beta=beta,
)
print("beta: {}".format(beta))


train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(
    dataset_name, dataset_fpath, batch_size, test_bs=batch_size
)

# 设置模型
if dataset_name == "cifar10":
    model_base = ModelFedCon("simple-cnn", out_dim=256, n_classes=10)
    glob_model_base = GlobModel("simple-cnn", out_dim=256, n_classes=10)
elif dataset_name == "cifar100":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=100)
    glob_model_base = GlobModel("resnet50-cifar100", out_dim=256, n_classes=100)

elif dataset_name == "tinyimagenet":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=200)
    glob_model_base = GlobModel("resnet50-cifar100", out_dim=256, n_classes=200)

shared_key_layers = list(
    set(model_base.state_dict().keys()) - set(glob_model_base.state_dict().keys())
)

model_fpath = "./client_checkpoint"
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)

client_d = {"avg": Client, "moon": MOONClient, "prox": ProxClient}


def inin_single_client(client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name, dataset_fpath, batch_size, batch_size, net_dataidx_map[client_id]
    )

    return {
        "trainer": CCVRTrainer(model_, optim_, nn.CrossEntropyLoss(), device, False),
        "trainloader": trainloader,
        "testloader": test_dl,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": args.strategy_name,
        "strategy": FedCCVR(model_fpath, glob_model_base),
        "save": False,
        "log": False,
    }


if __name__ == "__main__":
    # 客户端数量，及每轮上传客户端数量
    k = int(client_numbers * args.frac)
    print("客户端总数: {}; 每轮上传客户端数量: {}".format(client_numbers, k))

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(client_id)
        client_item = client_d[base_strategy](c_conf)
        client_item.model_trainer.strategy = base_strategy
        client_lst.append(client_item)

    s_conf = {
        "Round": 100,
        "client_numbers": client_numbers,
        "model_fpath": model_fpath,
        "iid": iid,
        "dataset_name": dataset_name,
        "strategy": FedCCVR(model_fpath, glob_model_base),
        "strategy_name": args.strategy_name,
        "log_suffix": args.suffix,
        "client_lst": client_lst,
    }

    server_o = sc(conf=s_conf)
    server_o.max_workers = 1

    kwargs = {"device": device}

    for ri in range(s_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k, **kwargs)
