# coding: utf-8
import copy

import torch
import torch.optim as optim

from .Distiller import KDLoss


class MDDistiller:
    def __init__(self, glob_model, w_local, device) -> None:
        super().__init__()
        self.glob_model = glob_model
        if w_local:
            self.load_model(w_local)

        temperature = 2
        self.criterion = KDLoss(temperature)
        # criterion = trainer.criterion
        self.optimizer = optim.SGD(self.glob_model.parameters(), lr=1e-2)
        self.device = device

    def load_model(self, w_dict):
        model_base_dict = self.glob_model.state_dict()
        model_base_dict.update(w_dict)
        self.glob_model.load_state_dict(model_base_dict, strict=False)

    def run(self, data_glob_d, epoch=1):
        x_lst, logits_lst = data_glob_d["x_lst"], data_glob_d["logits_lst"]

        # 为降低通信成本，该训练应该放到服务器端学习，再发回给各个客户端训练后不同的模型。但为了方便实现，该步骤先放到客户端进行
        for _ in range(epoch):
            for x, logits in zip(x_lst, logits_lst):
                x, logits = x.to(self.device), logits.to(self.device)
                y = self.glob_model(x)
                loss = self.criterion(y, logits)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.glob_model.state_dict()

    def predict(self, w_local_lst, data_loader):
        client_lst = []
        for w_local in w_local_lst:
            self.load_model(w_local)
            client_lst.append(copy.deepcopy(self.glob_model))

        x_lst = []
        logits_lst = []
        for x, _ in data_loader:
            x = x.to(self.device)

            logits = 0
            for client_m in client_lst:
                with torch.no_grad():
                    logits += client_m(x)

            logits /= len(w_local_lst)
            logits_lst.append(logits.cpu())
            x_lst.append(x.cpu())
        return x_lst, logits_lst
