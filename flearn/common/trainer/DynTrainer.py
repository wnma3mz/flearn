# coding: utf-8
import torch
import torch.nn.functional as F

from .Trainer import Trainer


class DynTrainer(Trainer):
    """
    FedDyn

    References
    ----------
    .. [1] Acar D A E, Zhao Y, Matas R, et al. Federated learning based on dynamic regularization[C]//International Conference on Learning Representations. 2020.

    项目地址: https://github.com/AntixK/FedDyn/blob/17e42576880f5dbe9d7d47cf112f2d88760dca55/feddyn/_feddyn.py#L184-L202
    """

    def __init__(self, model, optimizer, criterion, device, display=True, alpha=0.01):
        super(DynTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.server_state_dict = {}

        # save client's gradient
        self.prev_grads = None
        for param in self.model.parameters():
            zero_grad = torch.zeros_like(param.view(-1))
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = zero_grad
            else:
                self.prev_grads = torch.cat((self.prev_grads, zero_grad), dim=0)

        self.alpha = alpha

    def fed_loss(self):
        if self.server_state_dict != {}:
            # Linear penalty
            curr_params = None
            for name, param in self.model.named_parameters():
                if not isinstance(curr_params, torch.Tensor):
                    curr_params = param.view(-1)
                else:
                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)
            # 权重x梯度，尽可能大
            lin_penalty = torch.sum(curr_params * self.prev_grads)

            # Quadratic Penalty, 全局模型与客户端模型尽可能小
            quad_penalty = 0.0
            for name, param in self.model.named_parameters():
                quad_penalty += F.mse_loss(
                    param, self.server_state_dict[name].to(self.device), reduction="sum"
                )

            return -lin_penalty + self.alpha / 2.0 * quad_penalty
        else:
            return 0

    def update_info(self):
        # update prev_grads
        self.prev_grads = None
        for param in self.model.parameters():
            real_grad = param.grad.view(-1).clone()
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = real_grad
            else:
                self.prev_grads = torch.cat((self.prev_grads, real_grad), dim=0)
