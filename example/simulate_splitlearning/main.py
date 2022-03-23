# coding: utf-8
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import ModelFedCon
from utils import get_dataloader, partition_data

from flearn.common import Logger, Trainer
from flearn.common.utils import get_free_gpu_id, setup_seed

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

parser = argparse.ArgumentParser(description="Please input conf")
parser.add_argument("--local_epoch", dest="local_epoch", default=1, type=int)
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

# 设置模型
if dataset_name == "cifar10":
    model_base = ModelFedCon("simple-cnn", out_dim=256, n_classes=10)
elif dataset_name == "cifar100":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=100)
elif dataset_name == "tinyimagenet":
    model_base = ModelFedCon("resnet50-cifar100", out_dim=256, n_classes=200)

model_fpath = "./ckpts{}".format(args.suffix)
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)
shared_key_layers = [
    "l1.weight",
    "l1.bias",
    "l2.weight",
    "l2.bias",
    "l3.weight",
    "l3.bias",
]


class MyTrainer(Trainer):
    def batch(self, data, target, is_train=False):
        target = torch.tensor(target, dtype=torch.long).to(self.device)
        data = data.to(self.device)
        # print(self.model.training)
        if is_train:
            self.model.train()
            with torch.enable_grad():
                return super().batch(data, target)
        else:
            return super().batch(data, target)


def inin_single_client(model_base, client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    trainloader, testloader, _, _ = get_dataloader(
        dataset_name, dataset_fpath, batch_size, batch_size, net_dataidx_map[client_id]
    )

    return {
        "trainer": MyTrainer(model_, optim_, nn.CrossEntropyLoss(), device, False),
        "trainloader": trainloader,
        # "testloader": testloader,
        "testloader": test_dl,
        "model_fname": "client{}_round_{}.pth".format(client_id, "{}"),
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "save": False,
        "log": False,
    }


def get_shared_weight(trainer):
    # 提取权重
    return {k: v.cpu() for k, v in trainer.weight.items() if k in shared_key_layers}


def load_shared_weight(trainer, shared_weight):
    # 当前客户端权重
    update_weight = trainer.weight
    update_weight.update(shared_weight)
    trainer.model.load_state_dict(update_weight)
    return trainer.model


if __name__ == "__main__":

    # 客户端数量，及每轮上传客户端数量
    client_numbers = 10

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
        conf_params = inin_single_client(model_base, client_id)
        client_lst.append(conf_params)

    Round, per_round_ep = 100, 10
    shared_weight = {}
    log_server = Logger(
        "[Server]round{}_clients{}_{}.log".format(Round, len(client_lst), args.suffix),
        level="info",
    )
    log_fmt = "Server; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f}; TestAcc: {:.4f};"
    for ri in range(Round):
        round_loss_lst, round_trainacc_lst, round_testacc_lst = [], [], []
        # 为贴近真实情况，每次训练的顺序不同。使用np.random.shuffle来实现
        trainloader_iter_lst = [
            (id_, iter(client["trainloader"])) for id_, client in enumerate(client_lst)
        ]
        while True:
            stop_flag = False
            # 随机选取一个客户端的trainloader的一个batch
            np.random.shuffle(trainloader_iter_lst)
            for id_, iter_loader in trainloader_iter_lst:
                # 考虑两种情况。1. 正常取出数据->直接跳出循环；2. 取不出数据（迭代结束）->找到下一个能够取出数据的客户端
                try:
                    x, y = next(iter_loader)
                    stop_flag = False
                    break
                except StopIteration:
                    stop_flag = True
                    # break

            # 若遍历完所有客户端，stop_flag=True，则表示此时所有客户端都无法取出数据。该轮结束
            if stop_flag == True:
                break
            # print(id_, end=",")

            # 载入上一个客户端的权重
            if shared_weight != {}:
                client_lst[id_]["trainer"].model = load_shared_weight(
                    client_lst[id_]["trainer"], shared_weight
                )

            # 训练
            trainloss, trainacc = client_lst[id_]["trainer"].batch(x, y, is_train=True)

            # 提取权重
            shared_weight = get_shared_weight(client_lst[id_]["trainer"])
            round_loss_lst.append(trainloss)
            round_trainacc_lst.append(trainacc)

        """
        # 随机选取一个客户端的trainloader
        np.random.shuffle(client_lst)
        for id_, client in enumerate(client_lst):
            # 载入上一个客户端的权重
            if shared_weight != {}:
                client["trainer"].model = load_shared_weight(
                    client["trainer"], shared_weight
                )
            # 训练
            trainloss, trainacc = client["trainer"].train(
                client["trainloader"], epochs=per_round_ep
            )

            # 提取权重
            shared_weight = get_shared_weight(client["trainer"])

            round_loss_lst.append(trainloss)
            round_trainacc_lst.append(trainacc)
        """

        # 测试
        for id_, client in enumerate(client_lst):
            testloss, testacc = client["trainer"].test(client["testloader"])
            round_testacc_lst.append(testacc)

        log_server.logger.info(
            log_fmt.format(
                ri,
                np.mean(round_loss_lst),
                np.mean(round_trainacc_lst),
                np.mean(round_testacc_lst),
            )
        )
