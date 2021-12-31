import os

import torch
import torch.nn as nn
import torch.optim as optim

from flearn.client.models import MLP, LeNet5, UNet3D, UnetVAE3D


class MyLoss(nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        """自定义损失函数.

        Args:
            Wt1 (torch.tensor):
            Wt0 (torch.tensor):
        Returns:
            None
        """
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0

    def forward(self, inputs, targets, phase):
        """损失函数计算
        Args:
            inputs  (torch.tensor):
            targets (torch.tensor):
            phase   (torch.tensor):
        Returns:
            float: Float 损失函数计算值
        """
        loss = -(
            self.Wt1[phase] * targets * inputs.log()
            + self.Wt0[phase] * (1 - targets) * (1 - inputs).log()
        )
        return loss


class Net(object):
    def __init__(self, model_fpath, init_model_name):
        """已有网络模型.

        Args:
            model_fpath     (str): 模型存储路径
            init_model_name (str): 模型存储名称
        Returns:
            None
        """
        self.init_model_name = os.path.join(
            os.path.dirname(model_fpath), init_model_name
        )

        self.criterion = nn.CrossEntropyLoss()
        # criterion = MyLoss(Wt1, Wt0)
        # criterion = unet_vae_loss
        # criterion = nn.BCELoss()
        # optimizer = torch.optim.Adam(net_local.parameters(), lr=LR)

    def get(self, net_arch):
        """载入网络模型.

        Args:
            net_arch     (str): 选取模型架构
        Returns:
            tuple: Tuple (torch.tensor 模型参数, bool 模型层是否为序列表示)
        """
        seq = False
        if net_arch == "MLP":
            # net_local = MLP(28 * 28, 10) # mnist
            net_local = MLP(3 * 224 * 224, 2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
        elif net_arch == "cnn":
            net_local = LeNet5(2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
        elif net_arch == "alexnet":
            from torchvision.models import alexnet

            net_local = alexnet(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "densenet":
            from torchvision.models import densenet121

            net_local = densenet121(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["classifier.weight", "classifier.bias"]
        elif net_arch == "googlenet":
            from torchvision.models import googlenet

            net_local = googlenet(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["fc.weight", "fc.bias"]
        elif net_arch == "inception_v3":
            from torchvision.models import inception_v3

            net_local = inception_v3(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["fc.weight", "fc.bias"]
        elif net_arch == "mnasnet0_5":
            from torchvision.models import mnasnet0_5

            net_local = mnasnet0_5(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "mobilenet_v2":
            from torchvision.models import mobilenet_v2

            net_local = mobilenet_v2(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "PreResNet":
            from torchvision.models import PreResNet

            net_local = PreResNet(20, num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["fc.weight", "fc.bias"]
        elif net_arch == "resnet18":
            from torchvision.models import resnet18

            # net_local = resnet18(num_classes=2)  # covid2019
            net_local = resnet18(num_classes=10)  # mnist
            net_local.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )  # mnist
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["fc.weight", "fc.bias"]
        elif net_arch == "shufflenet_v2_x1_0":
            from torchvision.models import shufflenet_v2_x1_0

            net_local = shufflenet_v2_x1_0(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            # shared_key_layers = ["fc.weight", "fc.bias"]
        elif net_arch == "squeezenet1_0":
            from torchvision.models import squeezenet1_0

            net_local = squeezenet1_0(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "vgg11":
            from torchvision.models import vgg11

            net_local = vgg11(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "fcn_resnet50":
            # 共享卷积层
            from torchvision.models import segmentation

            net_local = segmentation.fcn_resnet50(num_classes=2)  # covid2019
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["classifier"]
        elif net_arch == "unetvae":
            net_local = UnetVAE3D(
                input_shape=(160, 192, 128),
                in_channels=len(("t1", "t2", "flair", "t1ce")),
                out_channels=3,
                init_channels=16,
                p=0.2,
            )  # brast2018
            torch.save(net_local.state_dict(), self.init_model_name)
            seq = True  # shared_key_layers = ["vae_branch.vconv0"]
        else:
            net_local, self.init_model_name = "", ""
        self.optimizer = optim.SGD(net_local.parameters(), lr=1e-4, momentum=0.9)

        return net_local, seq
