## 联邦学习框架

![Pypi](https://img.shields.io/pypi/v/cfl)

### Quickstart

1. 下载最新的[release版本](https://github.com/wnma3mz/flearn/releases/latest) 并使用pip安装。或手动下载源码，在当前目录进行编译`python setup.py sdist bdist_wheel`。
2. 切换至运行目录`cd example/mnist_cifar/`
3. 运行`python main.py --strategy_name avg --dataset_name mnist dataset_fpath 数据集路径`

详细解释见`example/mnist_cifar`中的[README.md](https://github.com/wnma3mz/flearn/tree/master/example/mnist_cifar)

### 进阶1——复现LG-FedAVG

- 修改`Client.py`，以及如何配置共享层

见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/LG_reproduction)

### 进阶2——复现FedProx

- 修改训练器，以运用至更多任务与模型

见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/Prox)


### 支持策略

- [x] FedSGD
- [x] FedAVG
- [x] FedAVGM
- [x] FedBN
- [x] LG-FedAVG
- [x] FedOPT
- [x] FedPAV

split-learning可见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/split_learning)，尚存在loss爆炸问题。


### 框架图

![CFL](./imgs/CFL.png)

### 工作流

![CFL工作流](./imgs/CFL工作流.png)

