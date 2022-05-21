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

- 2016
  - [X] [FedSGD/FedAVG](), Google, [Communication-Efficient Learning of Deep Networksfrom Decentralized Data](https://arxiv.org/pdf/1602.05629), PMLR
- 2018
  - [X] [FedProx](https://github.com/litian96/FedProx), CMU, [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), MLSys
- 2019
  - [X] [FedAVGM](), MIT/Google, [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335)
  - [X] [FedMD](https://github.com/diogenes0319/FedMD_clean), Harvard, [FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/pdf/1910.03581), NIPS 
- 2020
  - [x] [SCAFFOLD](), Google, [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378), ICML
  - [X] [LG-FedAVG](https://github.com/pliang279/LG-FedAvg), CMU, [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/pdf/2001.01523), NIPS WorkShop
  - [X] [FedOPT](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadagrad.py)(非官方), Google, [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295), ICLR
  - [X] [FedPAV](https://github.com/cap-ntu/FedReID), NTU, [Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/pdf/2008.11560), ACM MM
  - [X] [FedDistill](https://github.com/zhuangdizhu/FedGen)(非官方), Oulu, [Federated Knowledge Distillation](https://arxiv.org/pdf/2011.02367)
  - [X] [FedMutual](), ZJU, [Federated Mutual Learning](https://arxiv.org/pdf/2006.16765)
  - [X] [FedDF(Ensemble Distillation)](https://github.com/epfml/federated-learning-public-code/), EPFL, [Ensemble Distillation for Robust Model Fusion in Federated Learning](https://arxiv.org/pdf/2006.07242), NIPS
  - [x] [FedPer](), Adobe Research, [Federated learning with personalization layers](https://arxiv.org/pdf/1912.00818)
  - [ ] [FedBE](https://github.com/hongyouc/FedBE), Ohio State, [FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning](https://arxiv.org/abs/2009.01974), ICLR
  - [ ] [FedNova](), CMU, [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481), NIPS
- 2021
  - [X] [FedBN](https://github.com/med-air/FedBN), Princeton, [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/pdf/2102.07623), ICLR
  - [X] [FedDyn](https://github.com/AntixK/FedDyn), Boston, [FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION](https://arxiv.org/pdf/2111.04263), ICLR
  - [X] [MOON](https://github.com/QinbinLi/MOON), NUS/Berkeley, [Model-Contrastive Federated Learning](https://arxiv.org/pdf/2103.16257.pdf), CVPR
  - [X] [CCVR](), NUS/Huawei, [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/pdf/2106.05001), NIPS
  - [X] [FedGen](https://github.com/zhuangdizhu/FedGen), Michigan State, [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/pdf/2105.10056), ICML
  - [ ] [FedDist](), Grenoble Alpes, [A Federated Learning Aggregation Algorithm for Pervasive Computing: Evaluation and Comparison](https://arxiv.org/pdf/2110.10223), PerCom
  - [x] [FedRep](https://github.com/lgcollins/FedRep/), University of Texas at Austin, [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/pdf/2102.07078), ICML
- 2022
  - [ ] [FedKD](https://github.com/wuch15/FedKD), Tsinghua University/Microsoft Research Asia, [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x), Nature Communications
  - [ ] [FedProto](https://github.com/yuetan031/fedproto), Australian Artificial Intelligence Institute, [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://www.aaai.org/AAAI22Papers/AAAI-6846.YueT.pdf), AAAI

split-learning可见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/split_learning)，尚存在loss爆炸问题。

### TODO

- [ ] IDA聚合方式。[Inverse Distance Aggregation for Federated Learning with Non-IID Data](https://arxiv.org/pdf/2008.07665)
- [ ] FedKD有待更新，测试
- [ ] 添加FedLSD论文

### 框架图

![CFL](./imgs/CFL.png)

### 工作流

1. 服务器(Server)发送**训练指令**至各个客户端(Client)进行训练 (Server->Comm(S)->Comm(C)->Client)；模拟实验时，(Server->Client)
2. Client根据配置好的训练器(Trainer)进行训练，训练完成后返回指令至Server
3. Server发送**上传指令**至Client，Client根据配置好的策略(Strategy)，准备好上传的参数并进行上传，即Server发送指令后收到Client上传的参数
4. Server根据预先配好的Strategy对参数进行聚合
5. Server发送**接收指令**至Client，此时把参数发回至Client，Client根据配置好的Strategy进行接收
6. 若Server继续发送**测试指令**至Client，Client还要对更新后的模型进行验证，并返回验证后的结果至Server。否则，Server直接进行验证

P.S.
- Trainer中一般是需要配置联邦的损失函数`fed_loss`，主要作用是为了防止灾难性遗忘
- Distiller可以看作`fed_loss`，也可以看作聚合策略的一种，所以可能会在`Strategy`进行调用优化模型参数
- Strategy其实可以分为Server和Client两个部分，其中Client有两个函数（上传和接收）。此处是将策略看作一个整体，即Client和Server都调用同一个Strategy