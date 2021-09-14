## Server 测试

由于对于大部分联邦学习算法，客户端模型完全一致，因此在本框架中重复在每个客户端中测试全局测试集浪费大量时间。因此本例中修改`Server`，使其在全量测试集上仅测试一次。

### 运行

```bash
# 使用FedPAV在CIFAR-10上进行训练，公开数据集使用CIFAR-100
python3 main.py --dataset_fpath /mnt/data-ssd/CIFAR --public_fpath /mnt/data-ssd/CIFAR
```

### main.py解释

- line 120-128: 修改服务器端的评估部分，使其仅测试一个客户端`line 124`。
