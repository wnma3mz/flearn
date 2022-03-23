# coding: utf-8
import argparse
import base64
import copy
import os
import random

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from models import LeNet5Client, LeNet5Server, ResNet_cifarClient, ResNet_cifarServer
from split_data import iid as iid_f
from split_data import noniid

from flearn.client.datasets import get_dataloader, get_datasets, get_split_loader
from flearn.common.utils import get_free_gpu_id, setup_seed

idx = get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")

parser = argparse.ArgumentParser(description="Please input conf")
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

args = parser.parse_args()
iid = args.iid
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置数据集
dataset_name = args.dataset_name
dataset_fpath = "/mnt/data-ssd/"
num_classes = 10
batch_size = 128
if "cifar" in dataset_name:
    dataset_fpath = os.path.join(dataset_fpath, "CIFAR")
trainset, testset = get_datasets(dataset_name, dataset_fpath)

# 设置模型
if dataset_name == "mnist":
    model_base = LeNet5Client(num_classes=num_classes)
    model_server = LeNet5Server(num_classes=num_classes)

elif "cifar" in dataset_name:
    model_base = ResNet_cifarClient(
        dataset=args.dataset_name,
        resnet_size=8,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
    )
    model_server = ResNet_cifarServer(
        dataset=args.dataset_name,
        resnet_size=8,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
    )


setup_seed(0)

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
        "model": model_,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim_,
        "trainloader": trainloader,
        "testloader": testloader,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "device": device,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "save": False,
        "display": False,
        "log": False,
    }


def client_forward(data, model_):
    client_output_tmp = model_(data)
    client_output = client_output_tmp.clone().detach().requires_grad_(True)
    return client_output_tmp, client_output


def client_backward(client_output_tmp, client_grad, optimizer_):
    optimizer_.zero_grad()
    client_output_tmp.backward(client_grad)
    optimizer_.step()


if __name__ == "__main__":
    # 客户端数量，及每轮上传客户端数量
    client_numbers = 20
    k = int(client_numbers * args.frac)

    print("切分数据集")
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
    _, glob_testloader = get_dataloader(trainset, testset, batch_size, pin_memory=True)
    print("初始化客户端")
    client_lst = []
    for client_id in range(client_numbers):
        conf_params = inin_single_client(
            client_id, trainloader_idx_lst, testloader_idx_lst
        )
        client_lst.append(conf_params)

    Round = 1000

    import concurrent.futures

    for ri in range(Round):
        print("Round {}:".format(ri))
        round_loss_lst, round_trainacc_lst, round_testacc_lst = [], [], []
        for count in range(25):
            for id_, client in enumerate(client_lst):
                model_ = client["model"]
                optimizer_ = client["optimizer"]
                trainloader = client["trainloader"]
                testloader = client["testloader"]
                # testloader = glob_testloader

                model_.to(device)
                model_.train()
                loop_loss, accuracy = [], []

                client_output_tmp_lst, target_lst, output_lst = [], [], []
                for i, (data, target) in enumerate(trainloader):
                    # 每个用户每次只取一个batch_size
                    if i < count:
                        continue
                    if i > count:
                        break
                    data = data.to(device)

                    client_output_tmp, client_output = client_forward(data, model_)

                    json_d = {
                        "target": target.clone().detach().long(),
                        "client_output": client_output,
                    }
                    torch.save(json_d, "client_{}.pt".format(id_))
                    path = requests.post(
                        "http://localhost:5000/server",
                        json={"path": "client_{}.pt".format(id_), "client_id": id_},
                    ).json()["path"]
                    get_result = torch.load(path)
                    client_grad, acc, loss = (
                        get_result["grads"],
                        get_result["acc"],
                        get_result["loss"],
                    )

                    accuracy.append(acc)
                    loop_loss.append(loss / len(trainloader))

                    client_backward(client_output_tmp, client_grad, optimizer_)

                    client["model"] = model_

                # test
                model_.eval()
                test_loop_loss, test_accuracy = [], []

                client_output_tmp_lst, target_lst, output_lst = [], [], []
                for data, target in testloader:
                    data = data.to(device)
                    with torch.no_grad():
                        client_output_tmp, client_output = client_forward(data, model_)

                        json_d = {
                            "target": target.clone().detach().long(),
                            "client_output": client_output,
                        }
                        torch.save(json_d, "client_{}.pt".format(id_))
                        path = requests.post(
                            "http://localhost:5000/server",
                            json={
                                "path": "client_{}.pt".format(id_),
                                "client_id": id_,
                                "is_train": False,
                            },
                        ).json()["path"]
                        get_result = torch.load(path)
                        acc, loss = get_result["acc"], get_result["loss"]

                    test_accuracy.append(acc)
                    test_loop_loss.append(loss / len(trainloader))

                round_loss_lst.append(np.sum(loop_loss)),
                round_trainacc_lst.append(
                    np.sum(accuracy) / len(trainloader.dataset) * 100
                )
                round_testacc_lst.append(
                    np.sum(test_accuracy) / len(testloader.dataset) * 100
                )
                # print(
                #     "client {} Loss: {:.4f} TrainAcc: {:.4f} TestAcc: {:.4f}".format(
                #         id_,
                #         np.sum(loop_loss),
                #         np.sum(accuracy) / len(trainloader.dataset) * 100,
                #         np.sum(test_accuracy) / len(testloader.dataset) * 100,
                #     )
                # )

            print(
                "Loss: {:.4f} TrainAcc: {:.4f} TestAcc: {:.4f}".format(
                    np.mean(round_loss_lst),
                    np.mean(round_trainacc_lst),
                    np.mean(round_testacc_lst),
                )
            )
