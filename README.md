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
   - [x] [FedProx](https://github.com/litian96/FedProx), CMU, [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), MLSys
- 2019
   - [X] [FedAVGM](), MIT/谷歌, [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335)
   - [X] [FedMD](https://github.com/diogenes0319/FedMD_clean), Harvard, [FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/pdf/1910.03581), NIPS WorkShop
- 2020
   - [X] [LG-FedAVG](https://github.com/pliang279/LG-FedAvg), CMU, [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/pdf/2001.01523), NIPS WorkShop
   - [X] [FedOPT](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadagrad.py)(非官方), Google, [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295), ICLR
   - [x] [FedPAV](https://github.com/cap-ntu/FedReID), NTU, [Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/pdf/2008.11560), ACM MM
   - [X] [FedDistill](https://github.com/zhuangdizhu/FedGen)(非官方), Oulu, [Federated Knowledge Distillation](https://arxiv.org/pdf/2011.02367) 
   - [X] [FedMutual](), ZJU, [Federated Mutual Learning](https://arxiv.org/pdf/2006.16765)
   - [X] [FedDF(Ensemble Distillation)](https://github.com/epfml/federated-learning-public-code/), EPFL, [Ensemble Distillation for Robust Model Fusion in Federated Learning](https://arxiv.org/pdf/2006.07242), NIPS
   - [ ] [FedBE](https://github.com/hongyouc/FedBE), Ohio State, [FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning](https://arxiv.org/abs/2009.01974), ICLR
  - [ ] [FedNova](), CMU, [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481), NIPS
- 2021
  - [x] [FedBN](https://github.com/med-air/FedBN), Princeton, [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/pdf/2102.07623), ICLR
  - [X] [FedDyn](https://github.com/AntixK/FedDyn), Boston, [FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION](https://arxiv.org/pdf/2111.04263), ICLR
  - [x] [MOON](https://github.com/QinbinLi/MOON), NUS/Berkeley, [Model-Contrastive Federated Learning](https://arxiv.org/pdf/2103.16257.pdf), CVPR
  - [x] [CCVR](), NUS/Huawei, [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/pdf/2106.05001), NIPS
  - [x] [FedGen](https://github.com/zhuangdizhu/FedGen), Michigan State, [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/pdf/2105.10056), ICML
  - [ ] [FedProto](https://github.com/yuetan031/fedproto), Australian Artificial Intelligence Institute, [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://www.aaai.org/AAAI22Papers/AAAI-6846.YueT.pdf), AAAI
  - [ ] [FedDist](), Grenoble Alpes, [A Federated Learning Aggregation Algorithm for Pervasive Computing: Evaluation and Comparison](https://arxiv.org/pdf/2110.10223), PerCom
- 2022
  - [ ] [FedKD](https://github.com/wuch15/FedKD), Tsinghua University/Microsoft Research Asia, [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x), Nature Communications

split-learning可见[README.md](https://github.com/wnma3mz/flearn/tree/master/example/split_learning)，尚存在loss爆炸问题。

### TODO

- [ ] 策略改成组件形式，可任意搭配
- [ ] IDA聚合方式。[Inverse Distance Aggregation for Federated Learning with Non-IID Data](https://arxiv.org/pdf/2008.07665)


### 框架图

![CFL](./imgs/CFL.png)

### 工作流

![CFL工作流](./imgs/CFL工作流.png)
