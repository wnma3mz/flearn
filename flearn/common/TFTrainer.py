# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# 显示训练/测试过程
def show_f(fn):
    def wrapper(self, loader):
        if self.display == True:
            with tqdm(loader, ncols=80, postfix="loss: *.****; acc: *.**") as t:
                return fn(self, t)
        return fn(self, loader)

    return wrapper


class TFTrainer:
    """
    每轮训练/测试时函数调用顺序
    train/test --> eval_model       训练时调用其他模型, 设定eval模式
        --> _iteration              每轮的迭代器, 将数据传输至GPU上
        --> batch --> forward       每个batch的操作->模型forward, 考虑数据加载方式, 模型输出不止一个变量, 保存forward产生的数据等情况
                  --> fed_loss      联邦学习的损失计算
                  --> update_info   存储训练中产生的特征、标签等信息, 以便上传至服务器端
                  --> metrics       评估模型训练的准确率
        --> clear_info              训练大于一轮的情况下, 只需要保存最后一轮的信息, 清空其余轮的信息
    """

    def __init__(self, model, optimizer, criterion, display=True):
        """模型训练器

        Args:
            model :       torchvision.models
                          模型

            optimizer :   torch.optim
                          优化器

            criterion :   torch.nn.modules.loss
                          损失函数

            display :     bool (default: `True`)
                          是否显示过程
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.display = display

        self.history_loss = []
        self.history_accuracy = []

        self.iter_loss_f = tf.keras.metrics.Mean()
        self.iter_accuracy_f = tf.keras.metrics.SparseCategoricalAccuracy()

    def fed_loss(self):
        """联邦学习中, 客户端可能需要自定义其他的损失函数"""
        return 0

    def update_info(self):
        """每次训练后的更新操作, 保存信息。如特征等"""
        pass

    def clear_info(self):
        """如果不是最后一轮, 则无需上传。需要对保存的信息进行清空"""
        pass

    def eval_model(self):
        """在训练时, 由于联邦学习算法可能引入其他模型来指导当前模型, 所以需要提前将其他模型转为eval模式"""
        pass

    @tf.function
    def batch(self, data, target):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=self.is_train)
            loss = self.criterion(target, predictions)
        if self.is_train:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
        return self.iter_loss_f(loss), self.iter_accuracy_f(target, predictions)

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
            iter_loss, iter_acc = self.batch(data, target)
            loop_accuracy.append(iter_acc)
            loop_loss.append(iter_loss)
            if self.display:
                loader.postfix = "loss: {:.4f}; acc: {:.2f}".format(
                    iter_loss.numpy(), iter_acc.numpy()
                )

        if len(loop_accuracy) == 0:
            raise SystemExit("no training")
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
        self.is_train = True
        for ep in range(1, epochs + 1):
            loss, accuracy = self._iteration(data_loader)
            self.history_loss.append(loss)
            self.history_accuracy.append(accuracy)

            if ep != epochs:
                self.clear_info()
        return loss, accuracy

    def test(self, data_loader):
        """模型测试的初始入口, 由于只有一轮, 所以不需要loop
        Args:
            data_loader :  torch.utils.data
                           测试集

        Returns:
            float : loss
                    损失值

            float : accuracy
                    准确率
        """
        self.is_train = False
        loss, accuracy = self._iteration(data_loader)
        return loss, accuracy

    def save(self, fpath):
        """保存模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        self.model.save_weights(fpath)

    def restore(self, fpath):
        """恢复模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        # self.model = tf.keras.models.load_model(fpath)
        self.model.load_weights(fpath)

    @property
    def lr(self):
        """当前模型的学习率"""
        return self.optimizer._decayed_lr(tf.float32).numpy()

    @property
    def weight(self):
        """当前模型的参数"""
        return self.model.get_weights()
