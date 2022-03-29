# coding: utf-8
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from models import LeNet5, LeNet5Cifar10
from ProxClient import ProxClient
from ProxTrainer import L2Trainer, MaxTrainer, ProxTrainer
from resnet import ResNet_cifar
from utils import get_dataloader, partition_data

from flearn.client import Client
from flearn.common import Trainer
from flearn.common.utils import get_free_gpu_id, setup_seed
from flearn.server import Communicator as sc
from flearn.server import Server

# python3 main.py --strategy_name avg --dataset_name cifar10 --dataset_fpath ~/data --suffix _avg
# python3 main.py --strategy_name prox --dataset_name cifar10 --dataset_fpath ~/data --suffix _prox
# python3 main.py --strategy_name max --dataset_name cifar10 --dataset_fpath ~/data --suffix _max
# python3 main.py --strategy_name l2 --dataset_name cifar10 --dataset_fpath ~/data --suffix _l2 不用


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
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 客户端数量，及每轮上传客户端数量
client_numbers = 10
k = int(client_numbers * args.frac)
print("客户端总数: {}; 每轮上传客户端数量: {}".format(client_numbers, k))

# 设置数据集
batch_size = 64
beta = 0.5  # 当且仅当 "noniid" 时，有效
partition = "homo" if iid == True else "noniid"
print("切分{}数据集, 切割方式: {}".format(dataset_name, partition))

suffix = args.suffix + "_beta{}_iid{}".format(beta, iid == True)

trainer_d = {
    "prox": ProxTrainer,
    "max": MaxTrainer,
    "l2": L2Trainer,
    "avg": Trainer,
}


# 设置模型
if dataset_name == "mnist":
    model_base = LeNet5(num_classes=10)
elif dataset_name == "cifar10":
    model_base = LeNet5Cifar10(num_classes=10)
elif dataset_name == "cifar100":
    model_base = ResNet_cifar(
        dataset=args.dataset_name,
        resnet_size=8,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
    )

model_fpath = "./ckpts{}".format(args.suffix)
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)


def inin_single_client(model_base, client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name,
        dataset_fpath,
        batch_size,
        batch_size,
        net_dataidx_map[client_id],
        num_workers=4,
    )

    c_trainer = trainer_d[args.strategy_name](model_, optim_, criterion, device, False)

    return {
        "trainer": c_trainer,
        "trainloader": trainloader,
        "testloader": testloader,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": "avg",
        "save": False,
        "log": False,
    }


if __name__ == "__main__":
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

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(model_base, client_id)
        if args.strategy_name == "prox":
            client_lst.append(ProxClient(c_conf))
        else:
            client_lst.append(Client(c_conf))

    s_conf = {"model_fpath": model_fpath, "strategy_name": "avg"}
    sc_conf = {
        "server": Server(s_conf),
        "Round": 100,
        "client_numbers": client_numbers,
        "iid": iid,
        "dataset_name": dataset_name,
        "log_suffix": args.suffix,
        "client_lst": client_lst,
    }
    server_o = sc(conf=sc_conf)
    server_o.max_workers = 1
    for ri in range(sc_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k)
