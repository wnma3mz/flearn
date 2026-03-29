# FedMoE (DP-MoE) 示例

该示例在 `flearn` 框架内实现了一个从 0 到 1 的 FedMoE 最小可运行版本：

- 模型解耦：`backbone`（共享）+ `expert`（客户端本地）
- 聚合规则：只聚合 `backbone` 参数，`expert` 不参与全局平均
- Router（服务端）：基于客户端历史 loss 的轻量策略更新，用于下一轮客户端选择
- 本地训练：任务损失 + `backbone` 一致性约束（MSE）

## 代码结构

- `main.py`: 训练入口
- `models.py`: Backbone + Expert 网络
- `fedmoe_strategy.py`: 只聚合 backbone 的策略
- `fedmoe_trainer.py`: 一致性约束训练器
- `fedmoe_client.py`: 携带 expert_id 的客户端
- `fedmoe_server.py`: 带 loss-aware router 的服务端
- `split_data.py`: 面向 FedMoE 的 Non-IID 数据切分

## 运行方式

```bash
cd example/FedMoE
uv run python main.py \
  --dataset_name mnist \
  --dataset_fpath /path/to/datasets \
  --rounds 20 \
  --num_clients 20 \
  --num_experts 5 \
  --frac 0.5 \
  --local_epoch 1
```

## 对比实验入口

FedAvg/FedMoE 的对比实验代码已独立到 `example/benchmark_suite/`，避免与算法实现目录混在一起。
可直接查看：

- `example/benchmark_suite/README.md`

## 说明

这是 MVP 版本，重点验证 FedMoE 的核心机制（解耦聚合 + 路由选择 + Non-IID 专家分工）。
后续可继续将 router 升级为基于输入特征的在线策略网络（Policy Gradient）。
