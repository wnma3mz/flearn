import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, make_response, request

from flearn.client.utils import get_free_gpu_id
from flearn.common.utils import setup_seed
from models import LeNet5Client, LeNet5Server, ResNet_cifarClient, ResNet_cifarServer

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
dataset_name = args.dataset_name
num_classes = 10
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


idx = get_free_gpu_id()
print("使用{}号GPU".format(idx))
if idx != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.current_device()
    torch.cuda._initialized = True
else:
    raise SystemError("No Free GPU Device")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_server.to(device)
model_server.train()
optim_server = optim.SGD(model_server.parameters(), lr=1e-1)
criterion_server = nn.CrossEntropyLoss()


setup_seed(0)


app = Flask(__name__)


@app.route("/server", methods=["GET", "POST"])
def server_forward_backward():
    post_data = request.data
    post_data = json.loads(post_data)
    is_train = post_data["is_train"] if "is_train" in post_data.keys() else True
    json_d = torch.load(post_data["path"])
    target, client_output = json_d["target"], json_d["client_output"]

    target = target.to(device)
    client_output = client_output.to(device)

    output = model_server(client_output)
    loss = criterion_server(output, target)

    if is_train:
        optim_server.zero_grad()
        loss.backward()
        optim_server.step()
        client_grad = client_output.grad.clone().detach()
        data = {
            "grads": client_grad,
            "acc": (output.data.max(1)[1] == target.data).sum().item(),
            "loss": loss.data.item(),
        }
        torch.save(data, "server_{}.pt".format(post_data["client_id"]))
        return jsonify({"path": "server_{}.pt".format(post_data["client_id"])})
    data = {
        "acc": (output.data.max(1)[1] == target.data).sum().item(),
        "loss": loss.data.item(),
    }
    torch.save(data, "server_{}.pt".format(post_data["client_id"]))
    return jsonify({"path": "server_{}.pt".format(post_data["client_id"])})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port="23123", debug=True)
    # app.run(host="127.0.0.1", port="23123")
    # app.run(host="0.0.0.0", port="5000", debug=True)
    app.run(host="0.0.0.0", port="5000")
