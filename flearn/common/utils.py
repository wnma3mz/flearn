# coding: utf-8
import random

import numpy as np
import torch

from .strategy import AVG, AVGM, BN, LG, LG_R, OPT, PAV, SGD


def init_strategy(strategy_name, custom_strategy, model_fpath, shared_key_layers=None):
    if custom_strategy != None:
        return custom_strategy
    strategy_name = strategy_name.lower()
    strategy_d = {
        "avg": AVG(model_fpath),
        "sgd": SGD(model_fpath),
        "opt": OPT(model_fpath),
        "avgm": AVGM(model_fpath),
        "bn": BN(model_fpath),
        "lg": LG(model_fpath, shared_key_layers),
        "pav": PAV(model_fpath, shared_key_layers),
        "lg_r": LG_R(model_fpath, shared_key_layers),
    }
    if strategy_name not in strategy_d.keys():
        raise SystemError("Please input valid strategy name or strategy object!")
    return strategy_d[strategy_name]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
