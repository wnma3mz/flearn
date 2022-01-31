# coding: utf-8
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client import Client, datasets
from flearn.client import utils as fl_utils
from flearn.common.strategy import LG
from flearn.server import Communicator as sc
from flearn.server import Server
from model import GlobModel, ModelFedCon
from MyClients import DistillClient, DynClient, LSDClient, MOONClient, ProxClient
from MyStrategys import CCVR, DF, Distill, Dyn
from MyTrainers import (
    AVGTrainer,
    CCVRTrainer,
    DistillTrainer,
    DynTrainer,
    LSDTrainer,
    MOONTrainer,
    ProxTrainer,
)
from utils import get_dataloader, partition_data

# 设置随机数种子
fl_utils.setup_seed(0)
idx = fl_utils.get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 命令行参数
parser = argparse.ArgumentParser(description="Please input strategy_name")
parser.add_argument("--strategy_name", dest="strategy_name")
parser.add_argument("--local_epoch", dest="local_epoch", default=10, type=int)
parser.add_argument("--frac", dest="frac", default=1, type=float)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument("--iid", dest="iid", action="store_true")
parser.add_argument("--ccvr", dest="ccvr", action="store_true")
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
base_strategy = args.strategy_name.lower()
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath

# 可运行策略
trainer_d = {
    "avg": AVGTrainer,
    "moon": MOONTrainer,
    "prox": ProxTrainer,
    "lsd": LSDTrainer,
    "dyn": DynTrainer,
    "distill": DistillTrainer,
    "lg": AVGTrainer,
    "df": AVGTrainer,
}

client_d = {
    "avg": Client,
    "moon": MOONClient,
    "prox": ProxClient,
    "lsd": LSDClient,
    "dyn": DynClient,
    "distill": DistillClient,
    "lg": Client,
    "df": Client,
}


model_fpath = "./client_checkpoint"
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)

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

# 设置训练集以及策略
trainer = trainer_d[base_strategy] if base_strategy in trainer_d.keys() else None

shared_key_layers = [
    "l1.weight",
    "l1.bias",
    "l2.weight",
    "l2.bias",
    "l3.weight",
    "l3.bias",
]
strategy_d = {
    "dyn": Dyn(model_fpath, copy.deepcopy(model_base).state_dict()),
    "distill": Distill(model_fpath),
    "df": DF(model_fpath, copy.deepcopy(model_base)),
    "lg": LG(model_fpath, shared_key_layers=shared_key_layers),
}

if base_strategy == "dyn":
    ccvr_strategy = CCVR(
        model_fpath, glob_model_base, base_strategy, h=model_base.state_dict()
    )
elif base_strategy == "lg":
    ccvr_strategy = CCVR(
        model_fpath, glob_model_base, base_strategy, shared_key_layers=shared_key_layers
    )
else:
    ccvr_strategy = CCVR(model_fpath, glob_model_base, base_strategy)


def inin_single_client(model_base, client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name, dataset_fpath, batch_size, batch_size, net_dataidx_map[client_id]
    )

    if args.ccvr:
        c_trainer = CCVRTrainer(
            model_, optim_, nn.CrossEntropyLoss(), device, False, strategy=base_strategy
        )
    else:
        c_trainer = trainer(model_, optim_, nn.CrossEntropyLoss(), device, False)

    return {
        "trainer": c_trainer,
        "trainloader": trainloader,
        # "testloader": testloader,
        "testloader": test_dl,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": base_strategy,
        "save": False,
        "log": False,
    }


if __name__ == "__main__":

    # 客户端数量，及每轮上传客户端数量
    client_numbers = 10
    k = int(client_numbers * args.frac)
    print("客户端总数: {}; 每轮上传客户端数量: {}".format(client_numbers, k))

    # 设置数据集
    batch_size = 64
    beta = 0.5  # 当且仅当 "noniid" 时，有效
    partition = "homo" if iid == True else "noniid"
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

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(model_base, client_id)
        if args.ccvr:
            c_conf["strategy"] = copy.deepcopy(ccvr_strategy)
        elif base_strategy in strategy_d.keys():
            c_conf["strategy"] = copy.deepcopy(strategy_d[base_strategy])
        client_lst.append(client_d[base_strategy](c_conf))

    s_conf = {"model_fpath": model_fpath, "strategy_name": base_strategy}
    if args.ccvr:
        s_conf["strategy"] = copy.deepcopy(ccvr_strategy)
    elif base_strategy in strategy_d.keys():
        s_conf["strategy"] = copy.deepcopy(strategy_d[base_strategy])

    sc_conf = {
        "server": Server(s_conf),
        "Round": 100,
        "client_numbers": client_numbers,
        "log_suffix": args.suffix,
        "dataset_name": dataset_name,
        "client_lst": client_lst,
    }

    server_o = sc(conf=sc_conf)
    # server_o.max_workers = min(20, client_numbers)
    server_o.max_workers = 1
    kwargs = {"device": device}

    # 随意选取，可替换为更合适的数据集；或者生成随机数，见_create_data_randomly
    trainset, testset = datasets.get_datasets("cifar100", dataset_fpath)
    _, glob_testloader = datasets.get_dataloader(trainset, testset, 100, num_workers=0)
    kwargs = {
        "lr": 1e-2,
        "T": 2,
        "epoch": 1,
        "method": "avg_logits",
        "kd_loader": glob_testloader,
        "device": device,
    }

    for ri in range(sc_conf["Round"]):
        if args.ccvr:
            loss, train_acc, test_acc = server_o.run(ri, k=k, **kwargs)
        else:
            loss, train_acc, test_acc = server_o.run(ri, k=k)
