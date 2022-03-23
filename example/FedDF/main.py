# coding: utf-8
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from FedDF import DF
from models import LeNet5
from resnet import ResNet_cifar
from split_data import iid as iid_f
from split_data import noniid

from flearn.client import Client
from flearn.client.datasets import get_dataloader, get_datasets, get_split_loader
from flearn.common import Trainer
from flearn.common.utils import get_free_gpu_id, setup_seed
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

parser = argparse.ArgumentParser(description="")
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
batch_size = 128
trainset, testset = get_datasets(dataset_name, dataset_fpath)
# 全局测试集，所有客户端模型在此数据集上进行测试取平均。
_, glob_testloader = get_dataloader(trainset, testset, 100, pin_memory=True)

# 设置模型
if dataset_name == "mnist":
    model_base = LeNet5(num_classes=num_classes)
elif "cifar" in dataset_name:
    model_base = ResNet_cifar(
        dataset=args.dataset_name,
        resnet_size=8,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
    )

strategy_name = "df"
strategy = DF(model_base, device)

model_fpath = "./ckpts{}".format(args.suffix)
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)


def inin_single_client(client_id, trainloader_idx_lst, testloader_idx_lst):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-1)

    trainloader, testloader = get_split_loader(
        trainset,
        testset,
        trainloader_idx_lst[client_id],
        testloader_idx_lst[client_id],
        batch_size,
        num_workers=0,
    )

    return {
        "trainer": Trainer(model_, optim_, nn.CrossEntropyLoss(), device, False),
        "trainloader": trainloader,
        "testloader": [testloader, glob_testloader],
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


pn_normalize = True


def create_data_randomly():
    # create pseudo_data and map to [0, 1].
    # cifar10
    pseudo_data = torch.randn((batch_size, 3, 32, 32), requires_grad=False)
    pseudo_data = (pseudo_data - torch.min(pseudo_data)) / (
        torch.max(pseudo_data) - torch.min(pseudo_data)
    )

    # map values to [-1, 1] if necessary.
    if pn_normalize:
        pseudo_data = (pseudo_data - 0.5) * 2
    return pseudo_data, [0] * batch_size


if __name__ == "__main__":

    # 客户端数量，及每轮上传客户端数量
    client_numbers = 5
    k = int(client_numbers * args.frac)
    print("客户端总数: {}; 每轮上传客户端数量: {}".format(client_numbers, k))

    print("切分{}数据集, 切割方式iid={}".format(dataset_name, iid))
    if iid == "True":
        trainloader_idx_lst = iid_f(trainset, client_numbers)
        testloader_idx_lst = iid_f(testset, client_numbers)
    else:
        shard_per_user = 2
        if dataset_name == "cifar100":
            shard_per_user = 20
        trainloader_idx_lst, rand_set_all = noniid(
            trainset, client_numbers, shard_per_user
        )
        testloader_idx_lst, rand_set_all = noniid(
            testset, client_numbers, shard_per_user, rand_set_all=rand_set_all
        )
        print("每个客户端标签数量: {}".format(shard_per_user))

    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        c_conf = inin_single_client(client_id, trainloader_idx_lst, testloader_idx_lst)
        client_lst.append(Client(c_conf))

    sc_conf = {
        "model_fpath": model_fpath,
        "strategy": DF(model_base, device),
        "strategy_name": copy.deepcopy(strategy),
    }
    s_conf = {
        "server": Server(sc_conf),
        "Round": 1000,
        "client_numbers": client_numbers,
        "iid": iid,
        "dataset_name": dataset_name,
        "log_suffix": args.suffix,
        "client_lst": client_lst,
    }
    server_o = sc(conf=s_conf)
    server_o.max_workers = 1

    # 随意选取，可替换为更合适的数据集；或者生成随机数，见_create_data_randomly
    trainset, testset = get_datasets("cifar100", dataset_fpath)
    _, glob_testloader = get_dataloader(trainset, testset, 100, num_workers=0)
    kwargs = {
        "lr": 1e-2,
        "T": 2,
        "epoch": 1,
        "method": "avg_logits",
        "kd_loader": glob_testloader,
    }
    for ri in range(sc_conf["Round"]):
        loss, train_acc, test_acc = server_o.run(ri, k=k, **kwargs)
