# coding: utf-8
import random

import numpy as np
import torch

from flearn.common.strategy import (
    AVG,
    AVGM,
    BN,
    DF,
    LG,
    LG_R,
    MD,
    OPT,
    PAV,
    SGD,
    Distill,
    Dyn,
)

__all__ = ["setup_strategy", "setup_seed", "get_free_gpu_id"]


def setup_strategy(strategy_name, custom_strategy, **strategy_p):
    """设置框架内已有的策略
    Args:
        strategy_name     :  str
                             策略名称

        custom_strategy   :  flearn.common.Strategy
                             自定义的策略

        shared_key_layers :  list or None
                             共享的层名称(LG相关策略的参数)

    Returns:
        flearn.common.Strategy
        策略
    """
    shared_key_layers = strategy_p.get("shared_key_layers", None)
    model_base = strategy_p.get("model_base", None)
    strategy_base = strategy_p.get("strategy_base", AVG())
    h = strategy_p.get("h", None)
    glob_model = strategy_p.get("glob_model", None)
    device = strategy_p.get("device", "cpu")

    strategy_name = strategy_name.lower()

    strategy_d = {
        "avg": AVG(),
        "avgm": AVGM(),
        "bn": BN(),
        "df": DF(model_base, strategy_base),
        "distill": Distill(),
        "dyn": Dyn(h),
        "lg": LG(shared_key_layers),
        "lg_r": LG_R(shared_key_layers),
        "md": MD(shared_key_layers, glob_model, device),
        "opt": OPT(),
        "pav": PAV(shared_key_layers),
        "sgd": SGD(),
    }
    if strategy_name not in strategy_d.keys():
        if custom_strategy != None:
            return custom_strategy
        else:
            raise SystemError("Please input valid strategy name or strategy object!")
    return strategy_d[strategy_name]


def setup_seed(seed):
    """设置随机种子，以便于复现实验"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


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
