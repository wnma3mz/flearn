# coding: utf-8
import copy

import torch
import torch.nn.functional as F
from flearn.client import Client, Trainer


class DynTrainer(Trainer):
    """
    参考文献：

    [1] Acar D A E, Zhao Y, Matas R, et al. Federated learning based on dynamic regularization[C]//International Conference on Learning Representations. 2020.

    项目地址: https://github.com/AntixK/FedDyn/blob/17e42576880f5dbe9d7d47cf112f2d88760dca55/feddyn/_feddyn.py#L184-L202
    """

    def __init__(self, model, optimizer, criterion, device, display=True):
        super(DynTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_model = copy.deepcopy(self.model)
        self.server_state_dict = self.server_model.state_dict()

        self.prev_grads = None
        for param in self.server_model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat(
                    (self.prev_grads, torch.zeros_like(param.view(-1))), dim=0
                )

        self.alpha = 0.01

    def fed_loss(self):
        if self.server_model != None:
            # Linear penalty
            curr_params = None
            for name, param in self.model.named_parameters():
                if not isinstance(curr_params, torch.Tensor):
                    curr_params = param.view(-1)
                else:
                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)
            lin_penalty = torch.sum(curr_params * self.prev_grads)

            # Quadratic Penalty
            quad_penalty = 0.0
            for name, param in self.model.named_parameters():
                quad_penalty += F.mse_loss(
                    param, self.server_state_dict[name], reduction="sum"
                )

            return -lin_penalty + self.alpha / 2.0 * quad_penalty
        else:
            return 0

    def batch(self, data, target):
        output = self.model(data)
        loss = self.criterion(output, target)

        if self.is_train:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            self.prev_grads = None
            for param in self.model.parameters():
                if not isinstance(self.prev_grads, torch.Tensor):
                    self.prev_grads = param.grad.view(-1).clone()
                else:
                    self.prev_grads = torch.cat(
                        (self.prev_grads, param.grad.view(-1).clone()), dim=0
                    )

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc


class DynClient(Client):
    def revice(self, i, glob_params):
        w_local = self.model_trainer.weight
        self.w_local_bak = copy.deepcopy(w_local)
        # decode
        w_glob_b = self.encrypt.decode(glob_params)
        # update
        update_model = self.strategy.client_revice(self.model_trainer, w_glob_b)
        if self.scheduler != None:
            self.scheduler.step()
        # self.model_trainer.model.load_state_dict(self.w_local_bak)
        self.model_trainer.model = update_model
        self.model_trainer.server_model = copy.deepcopy(update_model)
        self.model_trainer.server_model.eval()
        self.model_trainer.server_state_dict = (
            self.model_trainer.server_model.state_dict()
        )
        return {
            "code": 200,
            "msg": "Model update completed",
            "client_id": self.client_id,
            "round": str(i),
        }
