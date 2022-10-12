# coding: utf-8
import numpy as np

from flearn.common.utils import setup_seed
from flearn.server import Server

setup_seed(0)
if __name__ == "__main__":
    dataset_name = "cifar10"
    strategy_name = "avg"
    s_conf = {
        "Round": 1,
        "client_numbers": 1,
        "model_fpath": ".",
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
    }

    s_model = Server(s_conf)

    client_id_lst = np.random.randint(10, size=10)
    assert (s_model.active_client(client_id_lst, -1) == client_id_lst).all()

    k = np.random.randint(10)
    assert len(s_model.active_client(client_id_lst, k)) == k

    k = 0
    acc_lst = [[np.random.randint(10, size=10)] * 2] * 10
    assert s_model.mean_lst(k, acc_lst) == np.mean(acc_lst[0][k])

    min_acc = 12
    val_acc_lst = np.random.randint(20, size=10)
    assert (np.where(val_acc_lst > min_acc)[0] == s_model.drop_client(val_acc_lst, min_acc)).all()
