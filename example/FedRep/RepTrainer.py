# coding: utf-8
import copy

import torch

from flearn.common.trainer import Trainer


class RepTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        display=True,
        shared_key_layers=None,
        rep_ep=5,
    ):
        """
        @params shared_key_layers : 通信的全局参数
        @params rep_ep            : 训练head的轮数
        """
        super().__init__(model, optimizer, criterion, device, display)
        if shared_key_layers:
            print("[WARNING]: Trainer don't have shared_key_layers_lst")

        self.shared_key_layers = shared_key_layers
        self.rep_ep = rep_ep

    def train(self, data_loader, epochs=1):
        # 保存训练前的模型，以计算梯度与配合FedSGD。多占用了一份显存
        self.weight_o = copy.deepcopy(self.model).cpu().state_dict()
        self.eval_model()
        self.model.train()
        for ep in range(1, epochs + 1):
            if ep < self.rep_ep:
                # 前rep_ep轮，先训练Head
                train_backbone, train_head = False, True
                for name, param in self.model.named_parameters():
                    if name in self.shared_key_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                # 后epochs-rep_ep再训练Backbone
                train_backbone, train_head = True, False
                for name, param in self.model.named_parameters():
                    if name in self.shared_key_layers:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)
            self.history_loss.append(loss)
            self.history_accuracy.append(accuracy)

            if ep != epochs:
                self.clear_info()
        return loss, accuracy
