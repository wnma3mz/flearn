## FedProx

复现FedProx

### 运行

```bash
# 使用基于FedAVG使用Prox在MNIST上进行训练，同理可替换为SGD、AVGM、BN、OPT
python3 main.py --strategy_name avg --suffix prox --dataset_name mnist --dataset_fpath /mnt/data-ssd
```

### main.py

- line 23-30: 自动选择最空闲的GPU
- line114: 客户端使用自定义的FedProxTrainer训练器
- 

### FedProxTrainer.py

line 14-21: 增加服务器端模型与本地训练模型的损失，$\frac{\mu}{2}||w-w^t||^2$。

其中，$\mu$在 `init`函数中进行初始化，第一轮不存在 `server_model`，所以不计算第一轮的损失。
