# coding: utf-8
import copy
import os
import sys
import unittest

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

from flearn.server import Communicator

CUR_DIR = os.path.dirname(__file__)
FEDMOE_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "..", "example", "FedMoE"))
if FEDMOE_DIR not in sys.path:
    sys.path.append(FEDMOE_DIR)

from fedmoe_client import FedMoEClient
from fedmoe_server import FedMoEServer
from fedmoe_strategy import FedMoEStrategy
from fedmoe_trainer import FedMoETrainer
from models import BackboneExpertNet


def make_loader(seed, size=64, in_channels=1, classes=10, batch_size=16):
    if torch is None:
        raise unittest.SkipTest("torch is required for FedMoE smoke test")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(size, in_channels, 28, 28, generator=g)
    y = torch.randint(0, classes, (size,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


class TestFedMoESmoke(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_end_to_end_and_backbone_only_aggregation(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        base_model = BackboneExpertNet(in_channels=1, num_classes=10, hidden_dim=32)
        base_strategy = FedMoEStrategy()

        client_lst = []
        for cid in range(4):
            model = copy.deepcopy(base_model)
            optimizer = optim.SGD(model.parameters(), lr=0.05)
            trainer = FedMoETrainer(model, optimizer, nn.CrossEntropyLoss(), device, display=False, lambda_consistency=0.01)
            conf = {
                "trainer": trainer,
                "trainloader": make_loader(seed=cid),
                "testloader": make_loader(seed=100 + cid, size=32),
                "client_id": cid,
                "model_fpath": ".",
                "epoch": 1,
                "dataset_name": "dummy",
                "strategy_name": "fedmoe",
                "strategy": copy.deepcopy(base_strategy),
                "save": False,
                "log": False,
            }
            client_lst.append(FedMoEClient(conf, expert_id=cid % 2))

        server_conf = {"model_fpath": ".", "strategy_name": "fedmoe", "strategy": copy.deepcopy(base_strategy)}
        server = FedMoEServer(server_conf)
        comm = Communicator(
            {
                "server": server,
                "Round": 2,
                "client_numbers": 4,
                "dataset_name": "dummy",
                "log": False,
                "client_lst": client_lst,
            }
        )
        comm.max_workers = 2

        loss0, train_acc0, _ = comm.run(0, k=2)
        loss1, train_acc1, _ = comm.run(1, k=2)

        self.assertTrue(loss0 >= 0)
        self.assertTrue(loss1 >= 0)
        self.assertTrue(train_acc0 >= 0)
        self.assertTrue(train_acc1 >= 0)

        # Backbone parameters should be synchronized after receiving global update.
        c0_state = client_lst[0].trainer.model.state_dict()
        c1_state = client_lst[1].trainer.model.state_dict()
        for key in c0_state.keys():
            if key.startswith("backbone") or key.startswith("backbone_proj"):
                self.assertTrue(torch.allclose(c0_state[key], c1_state[key], atol=1e-6))

        # Expert parameters are local and should not be forcibly synchronized.
        expert_equal = torch.allclose(c0_state["expert.0.weight"], c1_state["expert.0.weight"], atol=1e-6)
        self.assertFalse(expert_equal)


if __name__ == "__main__":
    unittest.main()
