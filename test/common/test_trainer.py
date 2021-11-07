import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flearn.common import Trainer


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(10 * 10, 2)

    def forward(self, x):
        x = x.view(-1, 10 * 10)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MLP()
    optim_ = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    criterion = F.cross_entropy
    device = "cpu"

    display = False
    t = Trainer(model, optim_, criterion, device, display=display)

    input_ = torch.rand(size=(32, 10, 10))
    target = torch.randint(0, 2, size=(32,), dtype=torch.long)
    iter_loss, iter_acc = t.batch(input_, target)
    print(iter_loss, iter_acc)

    loader = ((input_, target), (input_, target))
    epoch_loss, epoch_acc = t.train(loader, 5)
    print(epoch_loss, epoch_acc)

    display = True
    t = Trainer(model, optim_, criterion, device, display=display)
    epoch_loss, epoch_acc = t._iteration(loader)
    print(epoch_loss, epoch_acc)
