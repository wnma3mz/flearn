# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm


# 显示训练/测试过程
def show_f(fn):
    def wrapper(self, loader):
        if self.display == True:
            with tqdm(loader, ncols=80, postfix="loss: *.****; acc: *.**") as t:
                return fn(self, t)
        return fn(self, loader)

    return wrapper


class Trainer:
    # this flag allows you to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, criterion, device, display=True):
        """模型训练器

        Args:
            model :       torchvision.models
                          模型

            optimizer :   torch.optim
                          优化器

            criterion :   torch.nn.modules.loss
                          损失函数

            device :      torch.device
                          指定gpu还是cpu

            display :     bool (default: `True`)
                          是否显示过程
        """
        self.model = model
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.optimizer = optimizer
        self.criterion = criterion
        self.display = display
        self.model.to(self.device)
        self.history_loss = []
        self.history_accuracy = []

    def fed_loss(self):
        """联邦学习中，客户端可能需要自定义其他的损失函数"""
        return 0

    def update_info(self):
        """每次训练后的更新操作，保存信息。如特征等"""
        pass

    def clear_info(self):
        """如果不是最后一轮，则无需上传。需要对保存的信息进行清空"""
        pass

    def eval_model(self):
        """在训练时，由于联邦学习算法可能引入其他模型来指导当前模型，所以需要提前将其他模型转为eval模式"""
        pass

    @staticmethod
    def metrics(output, target):
        return (output.data.max(1)[1] == target.data).sum().item() / len(target) * 100

    def forward(self, data, target):
        output = self.model(data)
        return output

    def batch(self, data, target):
        """训练/测试每个batch的数据

        Args:
            data   : torch.tensor
                     训练数据

            target : torch.tensor
                     训练标签

        Returns:
            float : iter_loss
                    对应batch的loss

            float : iter_acc
                    对应batch的accuracy
        """
        output = self.forward(data, target)
        loss = self.criterion(output, target)

        if self.model.training:
            loss += self.fed_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_info()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    @show_f
    def _iteration(self, loader):
        """模型训练/测试的入口函数, 控制输出显示

        Args:
            data_loader : torch.utils.data
                          数据集

        Returns:
            float :
                    每个epoch的loss取平均

            float :
                    每个epoch的accuracy取平均
        """
        loop_loss, loop_accuracy = [], []
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            iter_loss, iter_acc = self.batch(data, target)
            loop_accuracy.append(iter_acc)
            loop_loss.append(iter_loss)

            if self.display:
                loader.postfix = "loss: {:.4f}; acc: {:.2f}".format(iter_loss, iter_acc)
        return np.mean(loop_loss), np.mean(loop_accuracy)

    def train(self, data_loader, epochs=1):
        """模型训练的入口
        Args:
            data_loader :  torch.utils.data
                           训练集

            epochs :       int
                           本地训练轮数

        Returns:
            float :
                    最后一轮epoch的loss

            float :
                    最后一轮epoch的accuracy
        """
        self.eval_model()
        self.model.train()
        for ep in range(1, epochs + 1):
            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)
            self.history_loss.append(loss)
            self.history_accuracy.append(accuracy)

            if ep != epochs:
                self.clear_info()
        return loss, accuracy

    def test(self, data_loader):
        """模型测试的初始入口，由于只有一轮，所以不需要loop
        Args:
            data_loader :  torch.utils.data
                           测试集

        Returns:
            float : loss
                    损失值

            float : accuracy
                    准确率
        """
        self.model.eval()
        with torch.no_grad():
            loss, accuracy = self._iteration(data_loader)
        return loss, accuracy

    def save(self, fpath):
        """保存模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            fpath,
        )

    def restore(self, fpath):
        """恢复模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        saved_data = torch.load(fpath)
        self.model.load_state_dict(saved_data["model"])
        self.optimizer.load_state_dict(saved_data["optimizer"])

    @property
    def lr(self):
        """当前模型的学习率"""
        return self.optimizer.state_dict()["param_groups"][0]["lr"]
        # return self.optimizer.param_groups[0]["lr"]

    @property
    def weight(self):
        """当前模型的参数"""
        return self.model.state_dict()

    @property
    def grads(self):
        """当前模型的梯度"""
        d = self.weight
        for k, v in d.items():
            d[k] = v - self.model_o[k]
        return d

    def add_grad(self, grads):
        """权重更新"""
        d = self.weight
        for k, v in d.items():
            d[k] = v + grads[k]
        self.model.load_state_dict(d)
