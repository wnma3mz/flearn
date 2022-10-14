# coding: utf-8
import copy

import torch

from flearn.client import Client
from flearn.common.strategy import BaseEncrypt, Strategy
from flearn.common.trainer import Trainer


# https://github.com/lgcollins/FedRep/blob/main/main_scaffold.py
class Scaffold(Strategy):
    def __init__(self, w_glob, device, lr_g=0.1, alg="scaf", encrypt=BaseEncrypt()):
        self.encrypt = encrypt
        self.w_glob = w_glob
        self.device = device
        self.alg = alg
        self.lr_g = lr_g

    @staticmethod
    def to_cpu(d):
        return {k: v.cpu() for k, v in d.items()}

    def to_gpu(self, d):
        return {k: v.to(self.device) for k, v in d.items()}

    def client(self, trainer, agg_weight=1):
        lr = trainer.lr
        w_local = self.to_cpu(trainer.weight)
        curr_c, last_c = self.to_cpu(trainer.curr_c), self.to_cpu(trainer.last_c)
        count = trainer.num_update

        return {
            "params": w_local,
            "last_c": last_c,
            "curr_c": curr_c,
            "lr": lr,
            "count": count,
        }

    def server(self, ensemble_params_lst, round_):
        lr_lst = self.extract_lst(ensemble_params_lst, "lr")

        last_c = self.extract_lst(ensemble_params_lst, "last_c")[0]
        c_list = self.extract_lst(ensemble_params_lst, "curr_c")
        w_local_lst = self.extract_lst(ensemble_params_lst, "params")
        count_lst = self.extract_lst(ensemble_params_lst, "count")

        w_glob = self.w_glob
        w_glob_here = copy.deepcopy(w_glob)
        delta_c, delta_y = {}, {}
        for k in w_glob.keys():
            delta_c[k] = torch.zeros(w_glob[k].size())
            delta_y[k] = torch.zeros(w_glob[k].size())

        lr = lr_lst[0]  # 每个客户端的学习率默认相同
        num_users = len(w_local_lst)  # 上传的客户端数量

        for idx in range(num_users):
            if self.alg == "scaf":
                ci_new = {}
                for jj, k in enumerate(w_glob.keys()):
                    ci_new[k] = (
                        c_list[idx][k]
                        - last_c[k]
                        + torch.div(
                            (w_glob_here[k] - w_local_lst[idx][k]),
                            count_lst[idx] * lr_lst[idx],
                        )
                    )
                    delta_y[k] = delta_y[k] + w_local_lst[idx][k] - w_glob_here[k]
                    delta_c[k] = delta_c[k] + ci_new[k] - c_list[idx][k]
            else:
                ci_new = {}
                for jj, k in enumerate(w_glob.keys()):
                    ci_new[k] = (
                        c_list[idx][k]
                        - torch.div(last_c[k], count_lst[idx])
                        + torch.div((w_glob_here[k] - w_local_lst[idx][k]), 0.1)
                    )
                    delta_y[k] = delta_y[k] + w_glob_here[k] - w_local_lst[idx][k]

        for k in w_glob.keys():
            if self.alg == "scaf":
                w_glob[k] += torch.mul(delta_y[k], self.lr_g / num_users)
                last_c[k] += torch.div(delta_c[k], num_users)
            else:
                last_c[k] = torch.div(delta_y[k], num_users * lr)
                w_glob[k] -= torch.mul(last_c[k], self.lr_g * lr)

        self.w_glob = w_glob
        return {"w_glob": w_glob, "last_c": last_c}

    def client_receive(self, trainer, data_glob_d):
        w_glob = data_glob_d["w_glob"]
        last_c = self.to_gpu(data_glob_d["last_c"])
        return w_glob, last_c


class ScaffoldClient(Client):
    def receive(self, i, glob_params):
        data_glob_d = self.strategy.receive_processing(glob_params)

        # update
        w_update, last_c = self.strategy.client_receive(self.trainer, data_glob_d)
        if self.scheduler != None:
            self.scheduler.step()
        self.trainer.model.load_state_dict(w_update)

        self.trainer.last_c = last_c

        if self.save:
            self.trainer.save(self.agg_fpath)

        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }


class ScaffoldTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True):
        super().__init__(model, optimizer, criterion, device, display)

        """
        # 原
        c_list = []
        for user in range(client_numbers + 1):
            ci = {}
            for k in model_base.state_dict().keys():
                ci[k] = torch.zeros(model_base.state_dict()[k].size()).to(device)
            c_list.append(copy.deepcopy(ci))
        """
        # 每个客户端只需要维护自己客户端以及最后一个即可
        ci = {}
        for k in self.model.state_dict().keys():
            ci[k] = torch.zeros(self.model.state_dict()[k].size()).to(device)
        self.curr_c = copy.deepcopy(ci)
        self.last_c = copy.deepcopy(ci)

    def fed_loss(self):
        local_par_list = None
        dif = None
        for param in self.model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

        # c_list[idx]，对应客户端的c_list
        for k in self.last_c.keys():
            if not isinstance(dif, torch.Tensor):
                dif = (-self.curr_c[k] + self.last_c[k]).reshape(-1)
            else:
                dif = torch.cat((dif, (-self.curr_c[k] + self.last_c[k]).reshape(-1)), 0)
        loss_algo = torch.sum(local_par_list * dif)
        return loss_algo

    def batch(self, data, target):
        _, loss, iter_acc = self.forward(data, target)

        if self.model.training:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)

            self.optimizer.step()

            self.num_update += 1

        return loss.data.item(), iter_acc

    def train(self, data_loader, epochs=1):
        self.num_update = 0
        return super().train(data_loader, epochs)
