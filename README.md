## 联邦学习框架

![Pypi](https://img.shields.io/pypi/v/cfl)

### Quickstart

1. 下载最新的[release版本](https://github.com/wnma3mz/flearn/releases/latest) 并使用pip安装。或手动下载源码，在当前目录进行编译 `python setup.py sdist bdist_wheel`。
2. 切换至运行目录 `cd example/mnist_cifar/`
3. 运行 `python main.py --strategy_name avg --dataset_name mnist dataset_fpath 数据集路径`

详细解释见 `example/mnist_cifar`中的[README.md](https://github.com/wnma3mz/flearn/tree/master/example/mnist_cifar)

### 进阶1——复现LG-FedAVG

- 修改 `Client.py`，以及如何配置共享层

见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/LG_reproduction)

### 进阶2——复现FedProx

- 修改训练器，以运用至更多任务与模型

见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/Prox)

### 进阶3——复现FedPAV

- 修改客户端以及服务器端，以适用于FedPAV策略

见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/PAV_reproduction)

### 支持策略

- [X] FedSGD/FedAVG [论文](https://arxiv.org/pdf/1602.05629)
- [X] FedAVGM [论文](https://arxiv.org/pdf/1909.06335)
- [X] FedBN [论文](https://arxiv.org/pdf/2102.07623) [代码](https://github.com/med-air/FedBN)
- [X] LG-FedAVG [论文](https://arxiv.org/pdf/2001.01523) [代码](https://github.com/pliang279/LG-FedAvg)
- [X] FedOPT [论文](https://arxiv.org/pdf/2003.00295) [代码](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadagrad.py)(非官方)
- [x] FedPAV [论文](https://arxiv.org/pdf/2008.11560) [代码](https://github.com/cap-ntu/FedReID)
- [x] FedProx [论文](https://arxiv.org/abs/1812.06127) [代码](https://github.com/litian96/FedProx)
- [ ] 复现计划
  - [X] FedDistill [论文](https://arxiv.org/pdf/2011.02367) [代码](<https://github.com/zhuangdizhu/FedGen>)(非官方)
  - [X] FedDyn [论文](https://arxiv.org/pdf/2111.04263) [代码]( https://github.com/AntixK/FedDyn)
  - [X] FedMD [论文](https://arxiv.org/pdf/1910.03581) [代码](https://github.com/diogenes0319/FedMD_clean)
  - [X] FedMutual [论文](https://arxiv.org/pdf/2006.16765)
  - [X] MOON [论文](https://arxiv.org/pdf/2103.16257.pdf) [代码](https://github.com/QinbinLi/MOON)
  - [X] CCVR [论文](https://arxiv.org/pdf/2106.05001) 
  - [X] FedGen [论文](arXiv preprint arXiv:2105.10056, 2021.) [代码](https://github.com/zhuangdizhu/FedGen)
  - [X] FedDF(Ensemble Distillation) [论文](https://arxiv.org/pdf/2006.07242) [代码](<https://github.com/epfml/federated-learning-public-code/>)
  - [ ] FedBE [论文](https://arxiv.org/abs/2009.01974) [代码](https://github.com/hongyouc/FedBE)
  - [ ] FedNova [论文](https://arxiv.org/pdf/2007.07481)
  - [ ] FedDist [论文](https://arxiv.org/pdf/2110.10223)

split-learning可见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/split_learning)，尚存在loss爆炸问题。

### TODO

- [ ] 策略改成组件形式，可任意搭配

### 框架图

![CFL](./imgs/CFL.png)

### 工作流

![CFL工作流](./imgs/CFL工作流.png)
