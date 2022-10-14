# coding: utf-8
import unittest

import numpy as np

from flearn.common.utils import setup_seed
from flearn.server import Server

setup_seed(0)


class TestServer(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        dataset_name = "cifar10"
        strategy_name = "avg"
        s_conf = {
            "Round": 1,
            "client_numbers": 1,
            "model_fpath": ".",
            "dataset_name": dataset_name,
            "strategy_name": strategy_name,
        }

        self.s = Server(s_conf)

    def test_func(self):
        client_id_lst = np.random.randint(10, size=10)
        np.testing.assert_array_equal(self.s.active_client(client_id_lst, -1), client_id_lst)

        k = np.random.randint(10)
        self.assertEqual(len(self.s.active_client(client_id_lst, k)), k)

        k = 0
        acc_lst = [[np.random.randint(10, size=10)] * 2] * 10
        self.assertEqual(self.s.mean_lst(k, acc_lst), np.mean(acc_lst[0][k]))

        min_acc = 12
        val_acc_lst = np.random.randint(20, size=10)
        data_lst = []
        for idx, val_acc in enumerate(val_acc_lst):
            data_lst.append({"client_id": idx, "val_acc": val_acc})

        np.testing.assert_array_equal(np.where(val_acc_lst > min_acc)[0], self.s.drop_client(data_lst, min_acc))


if __name__ == "__main__":

    t = TestServer("test_func")
    t.test_func()
