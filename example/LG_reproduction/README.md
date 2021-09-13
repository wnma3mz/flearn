## 基于FLearn复现LG-FedAVG

原项目：https://github.com/pliang279/LG-FedAvg/tree/master



本地快速切割MNIST/CIFAR-10/CIAFR-100数据集，模拟联邦学习设置

### 运行

```bash
# 使用LG-FedAVG在MNIST上进行训练
python3 main.py --strategy_name lg --frac 0.1 --dataset_name mnist --dataset_fpath /mnt/data-ssd
```

### 目录结构

```bash
├── client_checkpoint       # 客户端模型存储
├── main.py                 # 运行文件
├── models.py               # 模型文件，MLP, CNN-MNIST, CNN-CIFAR
└── split_data.py           # 切割数据集，IID与non-IID切割
└── LGClient.py             # 复写Client，每次载入最佳的模型
```

### 参数配置

- 策略选择`lg`或者`lg_r`，二者的区别在于共享模型的哪些层。`lg`表示选择共享给定的层，而`lg_r`表示**不共享**给定的层。默认使用`lg`即可。
- 客户端数量为100个，每轮上传10个；batch_size为50；MNIST与CIFAR-10的学习率分别为0.05与0.1；共享模型层见`shared_key_layers`参数。
- 不同于FedAVG，每个客户端模型存在差异。论文提出一种新的测试标准，每个客户端在全局测试集（`glob_testloader`）上进行测试，取平均值。对应的客户端配置中，`testloader`使用了一个列表，包含本地测试集与全局测试集。
- 一个小trick，客户端保存最好的模型，如果更新后的模型**不如**更新前的模型，则不进行更新。因此，需要额外复写`Client`，见`LGClient.py`。
- 由于客户端数量较多，测试时间较久，所有可以从第N轮再开始测试，见`LGClient.py`中的注释部分。

### main.py解释

- line 22-29: 自动选择最空闲的GPU
- line 120, 164: 配置共享模型中的部分层
- line 164: 由于客户端数量较多，因此改为单个线程进行，即串行执行。
