# coding: utf-8
import json
import numpy as np
import torch

listed_keys = [
    "model",
    "trainloader",
    "testloader",
    "valloader",
    "client_id",
    "optimizer",
    "criterion",
    "epoch",
    "model_fpath",
    "model_fname",
    "save",
    "restore_path",
    "trainer",
    "device",
    "display",
    "scheduler",
    "log",
    "log_suffix",
    "strategy_name",
    "strategy",
    "avg_round",
    "shared_key_layers",
    "dataset_name",
]
bool_key_lst = ["save", "display", "log"]
str_key_lst = [
    "restore_path",
    "trainer",
    "scheduler",
    "log_suffix",
    "strategy",
    "avg_round",
    "valloader",
    "shared_key_layers",
    "log_name_fmt",
]


def load_client_conf(
    model,
    criterion,
    optimizer,
    trainloader,
    testloader,
    fpath=None,
    device=None,
    model_fname=None,
    **kwargs
):
    """获取客户端配置.

    Args:
        model :         torchvision.models
                        模型

        criterion :     torch.nn.modules.loss
                        损失函数

        optimizer :     torch.optim
                        优化器

        trainloader :   torch.utils.data
                        训练数据集

        testloader :    torch.utils.data
                        测试数据集

        fpath :         str
                        客户端模型超参数配置文件路径

        device :        torch.device
                        使用GPU还是CPU

        model_fname :   str
                        每轮存储模型的名称

        kwargs :        dict
                        额外的配置参数 {"save": `False`, "display": `False`, "scheduler": ""}

    Returns:
        dict: Dict {
            "model_fpath" :       str
                                  客户端模型存储路径,

            "trainloader" :       torch.utils.data
                                  训练集,

            "testloader" :        torch.utils.data
                                  测试集,

            "optimizer" :         torch.optim
                                  优化器

            "criterion" :         torch.nn.modules.loss
                                  损失函数

            "shared_key_layers" : list
                                  共享网络中哪些层, ["fc.weight", "fc.bias"]

            "model" :             torchvision.models
                                  模型
        }
    """
    conf = dict()
    if fpath != None:
        with open(fpath, "r", encoding="utf-8") as f:
            conf = json.loads(f.read())

    # 覆盖文件中的属性
    for k, v in kwargs.items():
        conf[k] = v

    if trainloader == "" or testloader == "":
        print("datasets error!")
        assert trainloader != ""
        assert testloader != ""

    if model_fname == None:
        model_fname = "client{}_round_{}.pth".format(conf["client_id"], "{}")
    conf["model_fname"] = model_fname

    conf["trainloader"] = trainloader
    conf["testloader"] = testloader
    conf["optimizer"] = optimizer
    conf["criterion"] = criterion

    if "seq" in conf.keys() and conf["seq"]:
        w_local = model.state_dict()
        new_shared_key_layers = []
        for share_layer in conf["shared_key_layers"]:
            for layer in w_local.keys():
                if layer.startswith(share_layer):
                    new_shared_key_layers.append(layer)
        conf["shared_key_layers"] = new_shared_key_layers

    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf["device"] = device
    conf["model"] = model

    return conf


def get_free_gpu_id(num=1, min_memory=2000):
    """获取空闲GPU的ID.

    Args:
        num        : int
                     需要多少张GPU, default: 1

        min_memory : int
                     最小所需显存, 单位M, default: 2000

    Returns:
        list or int
    """
    import pynvml

    pynvml.nvmlInit()
    free_lst = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_lst.append(meminfo.free / 1024 / 1024)

    free_idx_lst = np.argsort(free_lst)[::-1][:num].tolist()
    for free_idx in free_idx_lst:
        if free_lst[free_idx] < min_memory:
            return -1

    if num == 1:
        return free_idx_lst[0]
    return free_idx_lst


if __name__ == "__main__":
    idx = get_free_gpu_id(num=1, min_memory=2000)
