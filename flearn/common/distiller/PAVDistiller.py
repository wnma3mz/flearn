# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PAVDistiller:
    def __init__(self, input_len, kd_loader, device, regularization=True):
        # input_len: 512
        self.input_len = input_len
        self.kd_loader = kd_loader
        self.device = device
        self.regularization = regularization

    def kd_generate_soft_label(self, model, data):
        """knowledge distillation (kd): generate soft labels."""
        with torch.no_grad():
            result = model(data)
        if self.regularization:
            # 对输出进行标准化
            result = F.normalize(result, dim=1, p=2)
        return result

    def run(self, teacher_lst, student):
        """
        teacher_lst: 客户端上传的模型参数
        student： 聚合后的模型
        kd_loader: 公开的大数据集
        注：这里的模型没有全连接层
        以每个客户端模型生成的label（平均）来教聚合后的模型
        """
        for teacher in teacher_lst:
            teacher.eval()
            teacher.to(self.device)
        student.train()
        student.to(self.device)
        MSEloss = nn.MSELoss().to(self.device)
        # lr=self.lr*0.01
        optimizer = optim.SGD(
            student.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )

        # kd_loader 公开的大数据集
        for _, (x, target) in enumerate(self.kd_loader):
            x, target = x.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            # 对应于模型全连接层的前一部分，512x10 or 512x100
            soft_target = torch.Tensor([[0] * self.input_len] * len(x)).to(self.device)

            for teacher in teacher_lst:
                soft_label = self.kd_generate_soft_label(teacher, x)
                soft_target += soft_label
            soft_target /= len(teacher_lst)

            output = student(x)

            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            # print("train_loss_fine_tuning", loss.data)
        return student
