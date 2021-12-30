# coding: utf-8
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client.utils import get_free_gpu_id
from flearn.server import Communicator as sc
from FML import FMLClient, FMLTrainer
from model import ModelFedCon
from utils import get_dataloader, partition_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


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


# 客户端数量，及每轮上传客户端数量
client_numbers = 10
k = int(client_numbers * args.frac)
print("客户端总数: {}; 每轮上传客户端数量: {}".format(client_numbers, k))

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
elif dataset_name == "cifar100":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=100)
elif dataset_name == "tinyimagenet":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=200)

model_fpath = "./client_checkpoint"
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)


def inin_single_client(client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name, dataset_fpath, batch_size, batch_size, net_dataidx_map[client_id]
    )
    return {
        "trainer": FMLTrainer(model_, optim_, nn.CrossEntropyLoss(), device, False),
        "trainloader": trainloader,
        # "testloader": testloader,
        "testloader": test_dl,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": args.strategy_name,
        "save": False,
        "log": False,
    }


if __name__ == "__main__":

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(client_id)
        client_lst.append(FMLClient(c_conf))

    s_conf = {
        "Round": 100,
        "client_numbers": client_numbers,
        "model_fpath": model_fpath,
        "iid": iid,
        "dataset_name": dataset_name,
        "strategy_name": args.strategy_name,
        "log_suffix": args.suffix,
        "client_lst": client_lst,
    }
    server_o = sc(conf=s_conf)
    server_o.max_workers = min(20, client_numbers)
    # server_o.max_workers = 20
    for ri in range(s_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k)
