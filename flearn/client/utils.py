# coding: utf-8
from dataclasses import dataclass
from typing import *

from flearn.common import Logger

listed_keys = [
    "trainloader",
    "testloader",
    "valloader",
    "client_id",
    "epoch",
    "model_fpath",
    "model_name_fmt",
    "save",
    "restore_path",
    "trainer",
    "scheduler",
    "log",
    "log_suffix",
    "strategy_name",
    "strategy",
    "avg_round",
    "shared_key_layers",
    "dataset_name",
]
bool_key_lst = ["save", "log", "save_best"]
str_key_lst = [
    "restore_path",
    "trainer",
    "scheduler",
    "strategy",
    "avg_round",
    "valloader",
    "shared_key_layers",
    "log_suffix",
    "log_name_fmt",
    "model_name_fmt",
]


@dataclass
class LogItem:
    log_suffix: str
    log_name_fmt: str
    log: bool


@dataclass
class DataInfo:
    label_distribute: Dict[int, int]  # 每个标签对应的样本数量
    num_class: int  # 训练集有多少个标签
    data_size: int  # 训练集总数


def init_log(log_name_fmt, client_id, dataset_name, log_suffix, strategy_name=None) -> Tuple[str, str]:
    """
    log_name_fmt    : 文件名
    client_id       : client唯一名称
    dataset_name    : 数据集名称
    log_suffix      : 文件名后缀
    strategy_name   : 策略名称，若为None，则表示SOLO训练
    """
    if log_name_fmt is None:
        log_name_fmt = "Client{}{}_dataset_{}{}.log"
    if log_suffix is None:
        log_suffix = ""
    strategy_name = "" if strategy_name is None else strategy_name

    log_client_name = log_name_fmt.format(client_id, "_" + strategy_name, dataset_name, log_suffix)
    log_client = Logger(log_client_name, level="info")
    log_fmt = client_id + "; Round: {}; Loss: {:.4f}; TrainAcc: {:.4f};"

    if strategy_name != "":
        log_fmt += "TestAcc: {};"
    else:
        log_fmt += "{}"
    return log_client, log_fmt
