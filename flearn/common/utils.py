# coding: utf-8
import random

import numpy as np
import torch

from .strategy import AVG, AVGM, BN, LG, LG_R, OPT, PAV, SGD


def init_strategy(strategy_name, custom_strategy, shared_key_layers=None):
    """初始化框架内已有的策略
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
    strategy_name = strategy_name.lower()
    strategy_d = {
        "avg": AVG(),
        "sgd": SGD(),
        "opt": OPT(),
        "avgm": AVGM(),
        "bn": BN(),
        "lg": LG(shared_key_layers),
        "pav": PAV(shared_key_layers),
        "lg_r": LG_R(shared_key_layers),
    }
    if strategy_name not in strategy_d.keys():
        if custom_strategy != None:
            return custom_strategy
        else:
            raise SystemError("Please input valid strategy name or strategy object!")
    return strategy_d[strategy_name]


def setup_seed(seed):
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
