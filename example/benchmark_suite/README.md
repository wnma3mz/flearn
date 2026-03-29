# Benchmark Suite

该目录用于统一放置 **FedAvg vs FedMoE** 对比实验代码，避免与 `example/FedMoE/` 的算法实现代码混在一起。

## 目标

- 多数据集对比（CPU 友好）
- 同一训练流程下比较不同方法
- 同一方法下比较不同模型架构（`cnn` / `mlp`）

## 单次运行

```bash
uv run python example/benchmark_suite/fedmoe_vs_fedavg_cpu.py \
  --dataset_name digits \
  --model_arch cnn \
  --param_budget same_hidden \
  --rounds 20 \
  --num_clients 10 \
  --frac 0.5
```

## 一次输出表格（多数据集 + 多模型）

```bash
uv run python example/benchmark_suite/fedmoe_vs_fedavg_table.py \
  --datasets digits,wine,iris,breast_cancer \
  --models cnn,mlp \
  --param_budget match_total \
  --seeds 1,2,3 \
  --rounds 20 \
  --num_clients 10 \
  --frac 0.5
```

参数预算模式说明：

- `same_hidden`: FedAvg 与 FedMoE 使用同样的 `hidden_dim`（默认）
- `match_total`: 自动放大 FedAvg 的 `hidden_dim`，使其总参数量接近 FedMoE 的唯一参数量（`backbone + num_clients * expert`）

表格输出会同时给出：

- 参数量对比（FedAvg / FedMoE unique）
- FLOPs/样本对比（FedAvg / FedMoE-Router）
- 多 seed 的 `mean ± std` 结果
