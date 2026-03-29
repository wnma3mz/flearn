# coding: utf-8
import argparse
import statistics
from types import SimpleNamespace

from fedmoe_vs_fedavg_cpu import run_benchmark


def dataset_labels_per_client(dataset_name):
    if dataset_name == "digits":
        return 2
    return 1


def mean_std(values):
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def fmt_pct(values):
    m, s = mean_std(values)
    return "{:.2f}% ± {:.2f}".format(m, s)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset + multi-model benchmark table")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--frac", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fedavg_hidden_dim", type=int, default=-1)
    parser.add_argument("--param_budget", type=str, default="same_hidden", choices=["same_hidden", "match_total"])
    parser.add_argument("--lambda_consistency", type=float, default=0.01)
    parser.add_argument("--router_epochs", type=int, default=5)
    parser.add_argument("--router_lr", type=float, default=1e-2)
    parser.add_argument("--router_max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--datasets", type=str, default="digits,wine,iris,breast_cancer")
    parser.add_argument("--models", type=str, default="cnn,mlp")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [args.seed]
    rows = []

    for model_arch in models:
        for dataset_name in datasets:
            results = []
            for seed in seeds:
                bench_args = SimpleNamespace(
                    dataset_name=dataset_name,
                    model_arch=model_arch,
                    rounds=args.rounds,
                    num_clients=args.num_clients,
                    labels_per_client=dataset_labels_per_client(dataset_name),
                    frac=args.frac,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    fedavg_hidden_dim=args.fedavg_hidden_dim,
                    param_budget=args.param_budget,
                    lambda_consistency=args.lambda_consistency,
                    router_epochs=args.router_epochs,
                    router_lr=args.router_lr,
                    router_max_samples=args.router_max_samples,
                    seed=seed,
                )
                results.append(run_benchmark(bench_args))

            first = results[0]
            rows.append(
                {
                    "model": model_arch,
                    "dataset": dataset_name,
                    "fedavg": [r["fedavg_final"] for r in results],
                    "oracle": [r["fedmoe_oracle_final"] for r in results],
                    "router": [r["fedmoe_router_final"] for r in results],
                    "router_top1": [r["router_top1_final"] for r in results],
                    "routing_gap": [r["routing_gap_final"] for r in results],
                    "router_gain": [r["gain_router_final"] for r in results],
                    "fedavg_params": first["fedavg_params"],
                    "fedmoe_params": first["fedmoe_unique_params"],
                    "fedavg_flops": first["fedavg_flops_per_sample"],
                    "fedmoe_router_flops": first["fedmoe_router_flops_per_sample"],
                    "fedavg_h": first["fedavg_hidden_dim"],
                    "fedmoe_h": first["fedmoe_hidden_dim"],
                }
            )

    print("## Multi-dataset, Multi-model CPU Benchmark")
    print()
    print("- Param budget mode: `{}`".format(args.param_budget))
    print("- Seeds: `{}`".format(",".join(str(x) for x in seeds)))
    print()
    print("### Table 1: Accuracy Comparison (Same Param Budget)")
    print()
    print(
        "| Model | Dataset | FedAvg (mean±std) | FedMoE-Router (mean±std) | Router Gain vs FedAvg (mean±std) |"
    )
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            "| {} | {} | {} | {} | {} |".format(
                row["model"],
                row["dataset"],
                fmt_pct(row["fedavg"]),
                fmt_pct(row["router"]),
                fmt_pct(row["router_gain"]),
            )
        )

    print()
    print("### Table 2: Method Comparison Under Same Param Budget")
    print()
    print(
        "| Model | Dataset | FedAvg Acc (mean±std) | FedMoE-Router Acc (mean±std) | Params (FedAvg / FedMoE) | Compute Ratio (FedMoE/FedAvg) |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        compute_ratio = row["fedmoe_router_flops"] / max(row["fedavg_flops"], 1)
        print(
            "| {} | {} | {} | {} | {:,} / {:,} | {:.3f}x |".format(
                row["model"],
                row["dataset"],
                fmt_pct(row["fedavg"]),
                fmt_pct(row["router"]),
                row["fedavg_params"],
                row["fedmoe_params"],
                compute_ratio,
            )
        )


if __name__ == "__main__":
    main()
