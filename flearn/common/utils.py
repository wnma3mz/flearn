# coding: utf-8
from .strategy import AVG, AVGM, BN, LG, LG_R, OPT, PAV, SGD


def init_strategy(strategy_name, model_fpath, shared_key_layers=None):
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
