# coding: utf-8

import numpy as np

listed_keys = [
    "trainloader",
    "testloader",
    "valloader",
    "client_id",
    "epoch",
    "model_fpath",
    "model_fname",
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
bool_key_lst = ["save", "log"]
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
