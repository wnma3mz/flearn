## MOON 复现

参考自：https://github.com/QinbinLi/MOON

### 运行

```bash
# 使用FedMOON在CIFAR10上进行训练
python3 main.py --strategy_name avg --suffix moon --dataset_name cifar10  --dataset_fpath /mnt/data-ssd
```

### 目录结构

```bash
├── FedMOON.py		# MOON的联邦设置
├── datasets.py		# 数据集读取
├── model.py		# MOON定义的模型
├── resnetcifar.py	# resnet模型
└── utils.py		# 数据集切割，Dirichlet distribution
```

### main.py解释

- line 
