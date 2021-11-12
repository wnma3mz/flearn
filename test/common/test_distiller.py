import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flearn.common.distiller import Distiller


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(10 * 10, 2)

    def forward(self, x):
        x = x.view(-1, 10 * 10)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    student = MLP()
    teacher = MLP()
    device = "cpu"

    input_ = torch.rand(size=(32, 10, 10))
    target = torch.randint(0, 2, size=(32,), dtype=torch.long)
    loader = ((input_, target), (input_, target))

    t = Distiller(loader, device)

    kwargs = {"lr": 0.01, "epoch": 1}
    t.single(teacher, student, **kwargs)

    teacher_lst = [MLP() for _ in range(3)]
    t.multi(teacher_lst, student, **kwargs)

    t.multi(teacher_lst, student, "avg_losses", **kwargs)

    t.multi(teacher_lst, student, "avg_probs", **kwargs)

