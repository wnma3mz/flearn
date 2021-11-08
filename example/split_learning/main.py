# coding: utf-8
import os

import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flearn.client.datasets import (
    get_datasets,
    get_split_loader,
    get_dataloader,
)
from flearn.client.utils import get_free_gpu_id
from models import LeNet5Server, LeNet5Client, ResNet_cifarServer, ResNet_cifarClient
from split_data import iid as iid_f
from split_data import noniid

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

optim_server = optim.SGD(model_server.parameters(), lr=1e-1)
criterion_server = nn.CrossEntropyLoss()

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


def server_forward_backward(target, client_output, device, is_train=True):
    target = target.to(device)
    client_output = client_output.to(device)

    output = model_server(client_output)
    loss = criterion_server(output, target)

    if is_train:
        optim_server.zero_grad()
        loss.backward()
        optim_server.step()
        client_grad = client_output.grad.clone().detach()
        return (
            client_grad,
            (output.data.max(1)[1] == target.data).sum().item(),
            loss.data.item(),
        )
    return (output.data.max(1)[1] == target.data).sum().item(), loss.data.item()


def client_forward(data, model_):
    client_output_tmp = model_(data)
    client_output = client_output_tmp.clone().detach().requires_grad_(True)
    return client_output_tmp, client_output


def client_backward(client_output_tmp, client_grad, optimizer_):
    optimizer_.zero_grad()
    client_output_tmp.backward(client_grad)
    optimizer_.step()


if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(0)

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
    model_server.to(device)
    model_server.train()
    for ri in range(Round):
        print("Round {}:".format(ri))
        round_loss_lst, round_trainacc_lst, round_testacc_lst = [], [], []
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
            for data, target in trainloader:
                data = data.to(device)

                client_output_tmp, client_output = client_forward(data, model_)

                client_grad, acc, loss = server_forward_backward(
                    target, client_output, device
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
                    acc, loss = server_forward_backward(
                        target, client_output, device, is_train=False
                    )

                test_accuracy.append(acc)
                test_loop_loss.append(loss / len(trainloader))

            round_loss_lst.append(np.sum(loop_loss)),
            round_trainacc_lst.append(np.sum(accuracy) / len(trainloader.dataset) * 100)
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
