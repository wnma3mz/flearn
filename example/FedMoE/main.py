# coding: utf-8
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client.datasets import get_datasets, get_split_loader
from flearn.common.utils import setup_seed
from flearn.server import Communicator as sc

from fedmoe_client import FedMoEClient
from fedmoe_server import FedMoEServer
from fedmoe_strategy import FedMoEStrategy
from fedmoe_trainer import FedMoETrainer
from models import BackboneExpertNet
from split_data import fedmoe_noniid

setup_seed(0)

parser = argparse.ArgumentParser(description="FedMoE training entry")
parser.add_argument("--local_epoch", dest="local_epoch", default=1, type=int)
parser.add_argument("--frac", dest="frac", default=0.5, type=float)
parser.add_argument("--rounds", dest="rounds", default=100, type=int)
parser.add_argument("--num_clients", dest="num_clients", default=20, type=int)
parser.add_argument("--num_experts", dest="num_experts", default=5, type=int)
parser.add_argument("--lambda_consistency", dest="lambda_consistency", default=0.01, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=128, type=int)
parser.add_argument("--suffix", dest="suffix", default="", type=str)
parser.add_argument(
    "--dataset_name",
    dest="dataset_name",
    default="mnist",
    choices=["mnist", "cifar10", "cifar100"],
    type=str,
)
parser.add_argument("--dataset_fpath", dest="dataset_fpath", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = args.dataset_name

in_channels = 1 if dataset_name == "mnist" else 3
num_classes = 100 if dataset_name == "cifar100" else 10

trainset, testset = get_datasets(dataset_name, args.dataset_fpath)
model_base = BackboneExpertNet(in_channels=in_channels, num_classes=num_classes, hidden_dim=128)

model_fpath = "./ckpts_fedmoe{}".format(args.suffix)
if not os.path.isdir(model_fpath):
    os.mkdir(model_fpath)

client_train_idx, client_expert_map = fedmoe_noniid(trainset, args.num_clients, args.num_experts)
client_test_idx, _ = fedmoe_noniid(testset, args.num_clients, args.num_experts)

base_strategy = FedMoEStrategy()


def init_single_client(client_id):
    model_ = copy.deepcopy(model_base)
    optim_ = optim.SGD(model_.parameters(), lr=1e-1, momentum=0.9)
    trainloader, testloader = get_split_loader(
        trainset,
        testset,
        client_train_idx[client_id],
        client_test_idx[client_id],
        args.batch_size,
        num_workers=0,
    )
    trainer = FedMoETrainer(
        model_,
        optim_,
        nn.CrossEntropyLoss(),
        device,
        display=False,
        lambda_consistency=args.lambda_consistency,
    )
    return {
        "trainer": trainer,
        "trainloader": trainloader,
        "testloader": testloader,
        "client_id": client_id,
        "model_fpath": model_fpath,
        "epoch": args.local_epoch,
        "dataset_name": dataset_name,
        "strategy_name": "fedmoe",
        "strategy": copy.deepcopy(base_strategy),
        "save": False,
        "log": False,
    }


if __name__ == "__main__":
    k = max(1, int(args.num_clients * args.frac))
    print("FedMoE | clients: {} | experts: {} | k: {}".format(args.num_clients, args.num_experts, k))

    client_lst = []
    for client_id in range(args.num_clients):
        c_conf = init_single_client(client_id)
        client_lst.append(FedMoEClient(c_conf, expert_id=client_expert_map[client_id]))

    s_conf = {"model_fpath": model_fpath, "strategy_name": "fedmoe", "strategy": copy.deepcopy(base_strategy)}
    server = FedMoEServer(s_conf)
    sc_conf = {
        "server": server,
        "Round": args.rounds,
        "client_numbers": args.num_clients,
        "dataset_name": dataset_name,
        "log_suffix": args.suffix,
        "client_lst": client_lst,
    }

    server_o = sc(conf=sc_conf)
    server_o.max_workers = min(20, args.num_clients)
    for ri in range(args.rounds):
        loss, train_acc, test_acc = server_o.run(ri, k=k)
        print(
            "Round {:03d} | loss {:.4f} | train_acc {:.2f} | test_acc {}".format(
                ri, loss, train_acc, test_acc
            )
        )
