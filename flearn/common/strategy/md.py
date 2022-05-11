# coding: utf-8
import copy

import torch
import torch.optim as optim

from flearn.common.strategy import LG_R


class MD(LG_R):
    def __init__(self, shared_key_layers, glob_model, device):
        super(MD, self).__init__(shared_key_layers)
        self.glob_model = glob_model
        if self.glob_model != None:
            self.device = device
            self.glob_model.to(device)
            self.glob_model.train()
        else:
            print("Warning: glob model is None")

    @staticmethod
    def load_model(model_base_dict, w_dict):
        model_base_dict.update(w_dict)
        return model_base_dict

    def client_revice(self, trainer, data_glob_d):
        w_local = trainer.weight
        w_local_bak = copy.deepcopy(w_local)
        self.glob_model.load_state_dict(
            self.load_model(self.glob_model.state_dict(), w_local)
        )
        # temperature = 2
        # criterion = KDLoss(temperature)
        criterion = trainer.criterion
        optimizer = optim.SGD(self.glob_model.parameters(), lr=1e-2)

        x_lst, logits_lst = data_glob_d["x_lst"], data_glob_d["logits_lst"]

        # 为降低通信成本，该训练应该放到服务器端学习，再发回给各个客户端训练后不同的模型。但为了方便实现，该步骤先放到客户端进行
        epoch = 1
        for _ in range(epoch):
            for x, logits in zip(x_lst, logits_lst):
                x, logits = x.to(self.device), logits.to(self.device)
                y = self.glob_model(x)
                loss = criterion(y, logits)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        w_glob_model = self.glob_model.state_dict()
        for k in w_glob_model.keys():
            w_local_bak[k] = w_glob_model[k]

        return w_local_bak

    def client_pub_predict(self, w_local_lst, **kwargs):
        data_loader = kwargs["data_loader"]

        client_lst = []
        for w_local in w_local_lst:
            self.glob_model.load_state_dict(
                self.load_model(self.glob_model.state_dict(), w_local)
            )
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

    def server(self, ensemble_params_lst, round_, **kwargs):
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        x_lst, logits_lst = self.client_pub_predict(w_local_lst, **kwargs)
        return {"x_lst": x_lst, "logits_lst": logits_lst, "w_glob": ""}
