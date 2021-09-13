## PAV reproduction

参考自：https://github.com/cap-ntu/FedReID/tree/master

### 运行

```bash
# 使用FedPAV在CIFAR-10上进行训练，公开数据集使用CIFAR-100
python3 main.py --dataset_fpath /mnt/data-ssd/CIFAR --public_fpath /mnt/data-ssd/CIFAR
```

### 目录结构

```bash
├── FedPAV.py 	 PAV的客户端与服务器端
```

### main.py解释

- line 37: public_fpath。公开数据集的路径，这里使用CIFAR-100，无标签数据集
- line 57-75: 除去基本的客户端模型配置外，还需要使用同样架构的服务器端模型。但由于上传部分没有最后一层，所以需要手动将`classifier`替换为`nn.Sequential()`。并且设定最后一层全连接层输入的大小`input_len`
- line 117, 162: 定义不参与共享的层，即`["classifier.weight", "classifier.bias"]`
- line 168-178: 服务器端的蒸馏部分，定义数据集、模型、gpu设备等。

### FedPAV.py解释

- line 12-28：增加蒸馏参数`kwargs`
- line 38-53：由于要计算cdw_feature的距离，所以需要额外定义变量。



- line 142-152: 服务器端设置。训练1000轮，`max_workers`控制线程数。默认用20个线程并行训练20个客户端模型。注：当使用多线程时，dataloader中的 `num_workers`必须为0。
