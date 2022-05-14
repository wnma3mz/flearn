import copy

import torch
import torch.nn as nn

from .Trainer import Trainer


class MOONTrainer(Trainer):
    """
    对应的网络结构如下所示，需要有proj层，返回h

    class ModelFedCon(nn.Module):
        def __init__(self, base_model, out_dim, n_classes, net_configs=None):
            super(ModelFedCon, self).__init__()

            self.features = SimpleCNN_header(
                input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes
            )
            num_ftrs = 84

            # projection MLP
            self.l1 = nn.Linear(num_ftrs, num_ftrs)
            self.l2 = nn.Linear(num_ftrs, out_dim)

            # last layer
            self.l3 = nn.Linear(out_dim, n_classes)

        def forward(self, x):
            h = self.features(x)
            # print("h before:", h)
            # print("h size:", h.size())
            h = h.squeeze()
            # print("h after:", h)
            x = self.l1(h)
            x = F.relu(x)
            x = self.l2(x)

            y = self.l3(x)
            return h, x, y

    """

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(MOONTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.global_model = copy.deepcopy(self.model)
        self.previous_model_lst = []
        self.cos = nn.CosineSimilarity(dim=-1)
        # CIFAR-10, CIFAR-100, and Tiny-Imagenet are 0.5, 1, and 0.5
        self.temperature = 0.5
        #  CIFAR-10, CIFAR-100, and Tiny-Imagenet are 5, 1, and 1
        self.mu = 5

    def fed_loss(self):
        if self.global_model != None:
            data, pro1 = self.data, self.pro1
            # 全局与本地的对比损失，越小越好
            with torch.no_grad():
                _, pro2, _ = self.global_model(data)
            posi = self.cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # 当前轮与上一轮的对比损失，越大越好
            for previous_net in self.previous_model_lst:
                with torch.no_grad():
                    _, pro3, _ = previous_net(data)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= self.temperature
            labels = torch.zeros(data.size(0)).to(self.device).long()

            return self.mu * self.criterion(logits, labels)
        else:
            return 0

    def eval_model(self):
        for previous_net in self.previous_model_lst:
            previous_net.eval()
            previous_net.to(self.device)

        if self.global_model != None:
            self.global_model.eval()
            self.global_model.to(self.device)

    def forward(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        _, pro1, output = self.model(data)
        self.data, self.pro1 = data, pro1

        loss = self.criterion(output, target)
        iter_acc = self.metrics(output, target)
        return output, loss, iter_acc
