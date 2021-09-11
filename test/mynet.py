import os
import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client import net


class MyNet(net):
    def __init__(self, model_fpath, init_model_name):
        super(MyNet, self).__init__(model_fpath, init_model_name)
        self.criterion = nn.CrossEntropyLoss()

    def get(self):
        seq = False
        # net_local = MLP(28 * 28, 10) # mnist
        net_local = MLP(3 * 224 * 224, 2)  # covid2019
        torch.save(net_local.state_dict(), self.init_model_name)
        self.optimizer = optim.SGD(net_local.parameters(), lr=1e-3, momentum=0.9)
        return net_local, seq
