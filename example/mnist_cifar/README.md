## Quickstart

本地快速切割MNIST/CIFAR-10/CIAFR-100数据集，模拟联邦学习设置

### 运行

```bash
# 使用FedAVG在MNIST上进行训练，同理可替换为SGD、AVGM、BN、OPT
python3 main.py --strategy_name avg --dataset_name mnist --dataset_fpath /mnt/data-ssd
```

### 目录结构

```bash
├── client_checkpoint       # 客户端模型存储
├── main.py                 # 运行文件
├── models.py               # 模型文件，LeNet-5
├── resnet.py               # 模型文件，resnet
└── split_data.py           # 切割数据集，IID与non-IID切割
```

### 参数配置

```bash
- strategy_name: 策略名称，如avg
- local_epoch:   每轮本地训练的epochs数，默认为1
- frac:          每轮上传客户端比例，默认为1.0
- suffix:        输出log文件后缀，默认为""
- iid:           是否为iid切割，默认为否
- dataset_name:  切割数据集名称
- dataset_fpath: 对应的数据集路径 
```

### main.py解释

- line 54-63: 根据数据集选用不同的模型。mnist使用LeNet-5模型，cifar数据集用ResNet-8模型。这里可直接切换为torchvision中的模型或自定义的模型
- line 79-108: 客户端参数初始化。其中line 83-90为获取切割后的训练集和测试集。每个客户端可根据各自的情况分别设置模型、数据集、优化器等。以下为部分配置参数的解释

  ```bash
  model_fname: 客户端模型存储的名称
  client_id:   客户端id名，默认为序号
  device:      使用哪块GPU
  model_fpath: 客户端模型存储路径
  save:        是否每轮都存储模型，默认为否。自动会存储在测试集上性能最好的
  display:     是否输出客户端训练进度条，单机情况下建议为否。
  log:         是否存储客户端训练日志
  ```
- line 122-132: 切割数据集。其中，当non-IID切割时，MNIST和CIFAR-10数据集每个客户端只有2类标签，而CIFAR-100的每个客户端有20个标签。
- line 142-152: 服务器端设置。训练1000轮，`max_workers`控制线程数。默认用20个线程并行训练20个客户端模型。注：当使用多线程时，dataloader中的 `num_workers`必须为0。
