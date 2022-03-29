## FedMAX

复现FedMAX


### 运行

```bash

```

https://github.com/weichennone/FedMAX/

### 结果

CIFAR-10数据集，beta=0.5, 10个客户端

|                    | Top-1 Acc |      |
| ------------------ | --------- | ---- |
| FedAVG             | 67.74     |      |
| FedProx_$\mu0.01$  | 67.07     |      |
| FedProx_$\mu1$     | 53.23     |      |
| FedMAX_$\beta1000$ | 64.99     |      |

