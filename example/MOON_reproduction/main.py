# coding: utf-8
import argparse
import copy
import os
from collections import defaultdict

import MyClients
import MyStrategys
import MyTrainers
import torch
import torch.nn as nn
import torch.optim as optim
from model import BackboneModel, HeadModel, ModelFedCon
from utils import get_dataloader, partition_data

from flearn.client import Client, DistillClient, MOONClient, ProxClient, datasets
from flearn.common.strategy import AVG
from flearn.common.trainer import MOONTrainer
from flearn.common.utils import get_free_gpu_id, setup_seed, setup_strategy
from flearn.server import Communicator as sc
from flearn.server import Server

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 命令行参数
parser = argparse.ArgumentParser(description="Please input strategy_name")
parser.add_argument("--strategy_name", dest="strategy_name")
parser.add_argument("--local_epoch", dest="local_epoch", default=10, type=int)
parser.add_argument("--frac", dest="frac", default=1, type=float)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument("--iid", dest="iid", action="store_true")
parser.add_argument("--ccvr", dest="ccvr", action="store_true")
parser.add_argument("--df", dest="df", action="store_true")
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
strategy_name = args.strategy_name.lower()
dataset_name = args.dataset_name
dataset_fpath = args.dataset_fpath
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

model_fpath = "./ckpts{}".format(suffix)
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)

# 设置模型
if dataset_name == "cifar10":
    model_base = ModelFedCon("simple-cnn", out_dim=256, n_classes=10)
    backbone_model_base = BackboneModel("simple-cnn", out_dim=256, n_classes=10)
    head_model_base = HeadModel("simple-cnn", out_dim=256, n_classes=10)
elif dataset_name == "cifar100":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=100)
    backbone_model_base = BackboneModel("resnet50-cifar100", out_dim=256, n_classes=100)
    head_model_base = HeadModel("resnet50-cifar100", out_dim=256, n_classes=100)
elif dataset_name == "tinyimagenet":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=200)
    backbone_model_base = BackboneModel("resnet50-cifar100", out_dim=256, n_classes=200)
    head_model_base = HeadModel("resnet50-cifar100", out_dim=256, n_classes=200)
# 设置策略
shared_key_layers = [
    "l1.weight",
    "l1.bias",
    "l2.weight",
    "l2.bias",
    "l3.weight",
    "l3.bias",
]

# 默认情况为FedAVG策略，即除了输入dyn和distill外，都是FedAVG策略
custom_strategy_d = defaultdict(lambda: AVG())

if strategy_name == "dyn":
    strategy_p = {"h": model_base.state_dict()}
elif strategy_name == "md":
    strategy_p = {
        "shared_key_layers": shared_key_layers,
        "model_base": copy.deepcopy(backbone_model_base),
        "device": device,
    }
else:
    strategy_p = {"shared_key_layers": shared_key_layers}
strategy = setup_strategy(strategy_name, custom_strategy_d[strategy_name], **strategy_p)
if args.ccvr and args.df:
    strategy = MyStrategys.DFCCVR(model_base, head_model_base, strategy)
elif args.ccvr:
    strategy = MyStrategys.CCVR(head_model_base, strategy)
elif args.df:
    strategy = MyStrategys.MyDF(model_base, strategy)

# 设置 训练器-客户端
conf_d = {
    "avg": {"trainer": MyTrainers.AVGTrainer, "client": Client},
    "moon": {"trainer": MOONTrainer, "client": MOONClient},
    "prox": {"trainer": MyTrainers.MyProxTrainer, "client": ProxClient},
    "lsd": {"trainer": MyTrainers.LSDTrainer, "client": MyClients.LSDClient},
    "dyn": {"trainer": MyTrainers.MyDynTrainer, "client": ProxClient},
    "lg": {"trainer": MyTrainers.AVGTrainer, "client": Client},
    "md": {"trainer": MyTrainers.AVGTrainer, "client": Client},
    "lg_r": {"trainer": MyTrainers.AVGTrainer, "client": Client},
    "distill": {
        "trainer": MyTrainers.MyDistillTrainer,
        "client": DistillClient,
    },
}


def inin_single_client(model_base, client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name,
        dataset_fpath,
        batch_size,
        batch_size,
        net_dataidx_map[client_id],
        num_workers=4,
    )

    trainer = conf_d[strategy_name]["trainer"]
    c_trainer = trainer(model_, optim_, criterion, device, False)

    if args.ccvr:
        c_trainer = MyTrainers.CCVRTrainer(c_trainer)

    return {
        "trainer": c_trainer,
        "trainloader": trainloader,
        "testloader": testloader,  # 对应数据集的所有测试数据，未切割
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "strategy": copy.deepcopy(strategy),
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

    # train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(
    #     dataset_name, dataset_fpath, batch_size, test_bs=batch_size, num_workers=4
    # )

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(model_base, client_id)
        client_lst.append(conf_d[strategy_name]["client"](c_conf))

    s_conf = {
        "model_fpath": model_fpath,
        "strategy": copy.deepcopy(strategy),
        "strategy_name": strategy_name,
    }
    sc_conf = {
        "server": Server(s_conf),
        "Round": 100,
        "client_numbers": client_numbers,
        "log_suffix": suffix,
        "dataset_name": dataset_name,
        "client_lst": client_lst,
    }

    server_o = sc(conf=sc_conf)
    # server_o.max_workers = min(20, client_numbers)
    server_o.max_workers = 1
    kwargs = {"device": device}

    # 随意选取，可替换为更合适的数据集；或者生成随机数，见_create_data_randomly
    if args.df:
        trainset, testset = datasets.get_datasets("cifar100", dataset_fpath)
        _, glob_testloader = datasets.get_dataloader(
            trainset, testset, 100, num_workers=4
        )
        kwargs = {
            "lr": 1e-2,
            "T": 2,
            "epoch": 1,
            "method": "avg_logits",
            "kd_loader": glob_testloader,
            "device": device,
        }

    if strategy_name == "md":
        trainset, testset = datasets.get_datasets("cifar100", dataset_fpath)
        _, glob_testloader = datasets.get_dataloader(
            trainset, testset, 100, num_workers=4
        )
        kwargs["data_loader"] = glob_testloader

    for ri in range(sc_conf["Round"]):
        if args.ccvr or args.df or strategy_name == "md":
            loss, train_acc, test_acc = server_o.run(ri, k=k, **kwargs)
        else:
            loss, train_acc, test_acc = server_o.run(ri, k=k)
