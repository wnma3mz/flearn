# coding: utf-8
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flearn.client.datasets import get_dataloader, get_datasets, get_split_loader
from flearn.client.utils import get_free_gpu_id
from flearn.common.strategy import LG, LG_R
from flearn.server import Communicate as sc

from LGClient import LGClient
from models import MLP, CNNCifar, CNNMnist
from split_data import iid as iid_f
from split_data import noniid

# 自动选择空闲显存最大的GPU
idx = get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")

parser = argparse.ArgumentParser(description="Please input strategy_name")
parser.add_argument("--strategy_name", dest="strategy_name", choices=["lg", "lg_r"])
parser.add_argument("--local_epoch", dest="local_epoch", default=1, type=int)
parser.add_argument("--frac", dest="frac", default=1, type=float)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument("--iid", dest="iid", action="store_true")
parser.add_argument(
    "--dataset_name",
    dest="dataset_name",
    default="mnist",
    choices=["mnist", "cifar10", "cifar100"],
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

# 设置数据集
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath
num_classes = 10
batch_size = 50
trainset, testset = get_datasets(dataset_name, dataset_fpath)
# 全局测试集，所有客户端模型在此数据集上进行测试取平均。
_, glob_testloader = get_dataloader(trainset, testset, 100, pin_memory=True)

# 设置模型
if dataset_name == "mnist":
    model_base = MLP(dim_in=784, dim_hidden=256, dim_out=10)
    shared_key_layers_lst = model_base.weight_keys[2:]
elif "cifar" in dataset_name:
    model_base = CNNCifar(num_classes=10)
    shared_key_layers_lst = model_base.weight_keys[3:]

shared_key_layers = []
for x in shared_key_layers_lst:
    shared_key_layers += x
print("共享层：", shared_key_layers)


model_fpath = "./client_checkpoint"
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


def inin_single_client(client_id, trainloader_idx_lst, testloader_idx_lst):
    model_ = copy.deepcopy(model_base)
    if dataset_name == "mnist":
        optim_ = optim.SGD(model_.parameters(), lr=0.05)
    elif "cifar" in dataset_name:
        optim_ = optim.SGD(model_.parameters(), lr=0.1)

    trainloader, testloader = get_split_loader(
        trainset,
        testset,
        trainloader_idx_lst[client_id],
        testloader_idx_lst[client_id],
        batch_size,
        num_workers=0,
    )

    return {
        "model": model_,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim_,
        "trainloader": trainloader,
        "testloader": [testloader, glob_testloader],
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "device": device,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": args.strategy_name,
        "shared_key_layers": shared_key_layers,
        "save": False,
        "display": False,
        "log": False,
    }


if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(0)

    # 客户端数量，及每轮上传客户端数量
    N_clients = 100
    k = int(N_clients * args.frac)
    print("客户端总数: {}; 每轮上传客户端数量: {}".format(N_clients, k))

    print("切分{}数据集, 切割方式iid={}".format(dataset_name, iid))
    if iid == "True":
        trainloader_idx_lst = iid_f(trainset, N_clients)
        testloader_idx_lst = iid_f(testset, N_clients)
    else:
        shard_per_user = 2
        if dataset_name == "cifar100":
            shard_per_user = 20
        trainloader_idx_lst, rand_set_all = noniid(trainset, N_clients, shard_per_user)
        testloader_idx_lst, rand_set_all = noniid(
            testset, N_clients, shard_per_user, rand_set_all=rand_set_all
        )
        print("每个客户端标签数量: {}".format(shard_per_user))

    print("初始化客户端")
    client_lst = []
    for client_id in range(N_clients):
        c_conf = inin_single_client(client_id, trainloader_idx_lst, testloader_idx_lst)
        client_lst.append(LGClient(c_conf))

    s_conf = {
        "Round": 200,
        "N_clients": N_clients,
        "model_fpath": model_fpath,
        "iid": iid,
        "dataset_name": dataset_name,
        "strategy_name": args.strategy_name,
        "shared_key_layers": shared_key_layers,
        "log_suffix": args.suffix,
    }
    server_o = sc(conf=s_conf, **{"client_lst": client_lst})
    server_o.max_workers = 1
    for ri in range(s_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k)