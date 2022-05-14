import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

tf_flag = False
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Flatten

    tf_flag = True if tf.__version__ > "2.0.0" else False
except:
    pass
tf_flag = False
if tf_flag:
    from flearn.common import TFTrainer

from flearn.common.trainer import Trainer
from flearn.common.utils import setup_seed

setup_seed(0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(10 * 10, 2)

    def forward(self, x):
        x = x.view(-1, 10 * 10)
        x = self.fc(x)
        return x


def tf_test():
    class MLP_TF(Model):
        def __init__(self):
            super(MLP_TF, self).__init__()
            self.flatten = Flatten()
            self.d1 = Dense(10 * 10)
            self.d2 = Dense(10)

        def call(self, x):
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    display = True
    batch_size = 128
    # trainloader, testloader = get_datasets("cifar10", batch_size)
    batch_loader = tf.random.uniform(shape=(32, 10, 10)), tf.random.uniform(
        shape=(32, 1)
    )
    trainloader = (batch_loader, batch_loader)
    optim_ = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = MLP_TF()
    t = TFTrainer(model, optim_, criterion, display=display)

    t.train(trainloader, 5)


if __name__ == "__main__":

    # tf
    if tf_flag:
        tf_test()

    # torch
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
