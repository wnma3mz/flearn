# coding: utf-8
import torch

from ..distiller import KDLoss
from .Trainer import Trainer


class LogitsTracker:
    """
    https://github.com/zhuangdizhu/FedGen/blob/HEAD/FLAlgorithms/users/userFedDistill.py
    """

    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.clear()

    def clear(self):
        # avoid division by zero error
        self.label_counts = torch.ones(self.unique_labels)
        self.logits_sums = torch.zeros((self.unique_labels, self.unique_labels))

    def update(self, logits, Y):
        """
        update logits tracker.
        :param logits: shape = n_samples * logits-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        logits, Y = logits.to("cpu"), Y.to("cpu")
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logits_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logits_sums_ = torch.zeros((self.unique_labels, self.unique_labels))
        logits_sums_.scatter_add_(0, labels, logits)
        self.logits_sums += logits_sums_

    def avg(self):
        return self.logits_sums.detach() / self.label_counts.float().unsqueeze(1)


class DistillTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        display=True,
        num_classes=10,
        kd_mu=1,
        temp=2,
    ):
        """
        num_classes: 10 # cifar10
        kd_mu:       1
        temp:        2
        """
        super(DistillTrainer, self).__init__(model, optimizer, criterion, device, display)
        self.logits_tracker = LogitsTracker(num_classes)  # cifar10
        self.glob_logits = None
        self.kd_mu = kd_mu
        self.kd_loss = KDLoss(temp)

    def fed_loss(self):
        if self.glob_logits != None:
            output, targets = self.output, self.targets.to(self.device)
            targets_p = self.glob_logits[targets, :]
            return self.kd_mu * self.kd_loss(output, targets_p)
        return 0

    def update_info(self):
        # 更新上传的logits
        self.logits_tracker.update(self.output, self.targets)

    def clear_info(self):
        self.logits_tracker.clear()

    def forward(self, data, targets):
        output, loss, iter_acc = super().forward(data, targets)
        self.output, self.targets = output, targets
        return output, loss, iter_acc
