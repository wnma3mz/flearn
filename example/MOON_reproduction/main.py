# coding: utf-8
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flearn.client import Client
from flearn.client.utils import get_free_gpu_id
from flearn.server import Communicator as sc

from FedMOON import AVGTrainer, MOONClient, MOONTrainer, ProxClient, ProxTrainer
from FedKD import LSDClient, LSDTrainer, SSDClient, DynClient, DynTrainer, Dyn

from model import ModelFedCon
from utils import get_dataloader, partition_data

idx = get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")

parser = argparse.ArgumentParser(description="Please input strategy_name")
parser.add_argument("--strategy_name", dest="strategy_name")
parser.add_argument("--local_epoch", dest="local_epoch", default=10, type=int)
parser.add_argument("--frac", dest="frac", default=1, type=float)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument("--iid", dest="iid", action="store_true")
parser.add_argument(
    "--dataset_name",
    dest="dataset_name",
    default="cifar10",
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
if args.strategy_name.lower() in ["moon", "lsd", "ssd", "lsdn", "prox", "dyn", "dane"]:
    strategy_name = "avg"
else:
    strategy_name = args.strategy_name.lower()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置数据集
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath
batch_size = 64
if iid == True:
    partition = "homo"
else:
    partition = "noniid"
    beta = 0.5
print("切分{}数据集, 切割方式: {}".format(dataset_name, partition))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)

# 客户端数量，及每轮上传客户端数量
N_clients = 10
k = int(N_clients * args.frac)
print("客户端总数: {}; 每轮上传客户端数量: {}".format(N_clients, k))

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
    n_parties=N_clients,
    beta=beta,
)
print("beta: {}".format(beta))


train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(
    dataset_name, dataset_fpath, batch_size, test_bs=batch_size
)

# 设置模型
if dataset_name == "cifar10":
    model_base = ModelFedCon("simple-cnn", out_dim=256, n_classes=10)
elif dataset_name == "cifar100":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=100)
elif dataset_name == "tinyimagenet":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=200)

model_fpath = "./client_checkpoint"
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)

trainer_d = {
    "avg": AVGTrainer,
    "moon": MOONTrainer,
    "prox": ProxTrainer,
    "lsd": LSDTrainer,
    "ssd": LSDTrainer,
    "lsdn": LSDTrainer,
    "dyn": DynTrainer,
}

trainer = None
s_name = args.strategy_name.lower()
if s_name in trainer_d.keys():
    trainer = trainer_d[s_name]


def inin_single_client(client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name, dataset_fpath, batch_size, batch_size, net_dataidx_map[client_id]
    )

    return {
        "model": model_,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim_,
        "trainloader": trainloader,
        # "testloader": testloader,
        "testloader": test_dl,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "device": device,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "trainer": trainer,
        "save": False,
        # "display": True,
        "display": False,
        "log": False,
    }


if __name__ == "__main__":

    print("初始化客户端")
    client_lst = []
    for client_id in range(N_clients):
        c_conf = inin_single_client(client_id)
        if args.strategy_name.lower() == "avg":
            client_lst.append(Client(c_conf))
        elif args.strategy_name.lower() == "moon":
            client_lst.append(MOONClient(c_conf))
        elif args.strategy_name.lower() == "prox":
            client_lst.append(ProxClient(c_conf))
        elif args.strategy_name.lower() == "lsd":
            client_lst.append(LSDClient(c_conf))
        elif args.strategy_name.lower() == "ssd":
            client_lst.append(SSDClient(c_conf))
        elif args.strategy_name.lower() == "dyn":
            c_conf["strategy"] = Dyn(
                model_fpath, copy.deepcopy(model_base).state_dict()
            )
            client_lst.append(DynClient(c_conf))
    s_conf = {
        "Round": 100,
        "N_clients": N_clients,
        "model_fpath": model_fpath,
        "iid": iid,
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "log_suffix": args.suffix,
    }
    if args.strategy_name.lower() == "dyn":
        s_conf["strategy"] = Dyn(model_fpath, copy.deepcopy(model_base).state_dict())
    # server_o = sc(conf=s_conf, Server=MOONServer, **{"client_lst": client_lst})
    server_o = sc(conf=s_conf, **{"client_lst": client_lst})
    # server_o.max_workers = min(20, N_clients)
    server_o.max_workers = 1
    for ri in range(s_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k)
