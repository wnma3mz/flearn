# coding: utf-8
import argparse
import copy
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import build_model


class RouterLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RouterLinear, self).__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def count_model_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_backbone_expert_params(model):
    backbone_params, expert_params = 0, 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name.startswith("backbone") or name.startswith("backbone_proj"):
            backbone_params += n
        elif name.startswith("expert"):
            expert_params += n
    return backbone_params, expert_params


def find_hidden_dim_for_target(model_arch, num_classes, target_params):
    low, high = 4, 16
    while True:
        model = build_model(model_arch, num_classes=num_classes, hidden_dim=high)
        if count_model_params(model) >= target_params or high >= 16384:
            break
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        model = build_model(model_arch, num_classes=num_classes, hidden_dim=mid)
        if count_model_params(model) < target_params:
            low = mid
        else:
            high = mid

    cand = []
    for h in [low, high]:
        model = build_model(model_arch, num_classes=num_classes, hidden_dim=h)
        cand.append((abs(count_model_params(model) - target_params), h))
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def estimate_flops_per_sample(model_arch, hidden_dim, num_classes, num_clients, include_router=False):
    """Rough FLOPs per sample (multiply-add counted as 2 FLOPs)."""
    model_arch = model_arch.lower()
    if model_arch == "cnn":
        # input 1x8x8
        conv1 = 2 * 8 * 8 * 16 * (1 * 3 * 3)
        conv2 = 2 * 4 * 4 * 32 * (16 * 3 * 3)
        proj = 2 * (32 * 2 * 2) * hidden_dim
        exp1 = 2 * hidden_dim * hidden_dim
        exp2 = 2 * hidden_dim * num_classes
        flops = conv1 + conv2 + proj + exp1 + exp2
    elif model_arch == "mlp":
        # input flattened as 64-dim vector
        l1 = 2 * 64 * hidden_dim
        proj = 2 * hidden_dim * hidden_dim
        exp1 = 2 * hidden_dim * hidden_dim
        exp2 = 2 * hidden_dim * num_classes
        flops = l1 + proj + exp1 + exp2
    else:
        raise ValueError("unsupported model_arch: {}".format(model_arch))

    if include_router:
        flops += 2 * hidden_dim * num_clients
    return int(flops)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_columns(x):
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    denom = np.where((x_max - x_min) < 1e-8, 1.0, (x_max - x_min))
    return (x - x_min) / denom


def _tabular_to_image8x8(x):
    n, d = x.shape
    out = np.zeros((n, 64), dtype=np.float32)
    out[:, : min(d, 64)] = x[:, :64]
    return out.reshape(n, 8, 8)


def load_dataset(dataset_name, test_size=0.3, seed=42):
    dataset_name = dataset_name.lower()
    if dataset_name == "digits":
        data = load_digits()
        x = data.images.astype(np.float32) / 16.0
        y = data.target.astype(np.int64)
    elif dataset_name == "wine":
        data = load_wine()
        x = _tabular_to_image8x8(_normalize_columns(data.data.astype(np.float32)))
        y = data.target.astype(np.int64)
    elif dataset_name == "iris":
        data = load_iris()
        x = _tabular_to_image8x8(_normalize_columns(data.data.astype(np.float32)))
        y = data.target.astype(np.int64)
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        x = _tabular_to_image8x8(_normalize_columns(data.data.astype(np.float32)))
        y = data.target.astype(np.int64)
    else:
        raise ValueError("unsupported dataset_name: {}".format(dataset_name))

    labels = sorted(np.unique(y).tolist())
    label_map = {label: i for i, label in enumerate(labels)}
    y = np.array([label_map[int(v)] for v in y], dtype=np.int64)
    num_classes = len(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    x_train = torch.tensor(x_train).unsqueeze(1)
    x_test = torch.tensor(x_test).unsqueeze(1)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    return x_train, y_train, x_test, y_test, num_classes


def build_noniid_partition(y, num_clients, labels_per_client=2, num_classes=10):
    labels_per_client = max(1, min(labels_per_client, num_classes))
    label_clients = defaultdict(list)
    for cid in range(num_clients):
        labels = [((cid * labels_per_client) + i) % num_classes for i in range(labels_per_client)]
        for label in labels:
            label_clients[label].append(cid)

    label_indices = defaultdict(list)
    y_np = y.numpy()
    for idx, label in enumerate(y_np):
        label_indices[int(label)].append(idx)
    for label in range(num_classes):
        random.shuffle(label_indices[label])

    client_indices = {cid: [] for cid in range(num_clients)}
    for label, indices in label_indices.items():
        owners = label_clients[label]
        chunks = np.array_split(indices, len(owners))
        for cid, chunk in zip(owners, chunks):
            client_indices[cid].extend(chunk.tolist())

    for cid in range(num_clients):
        random.shuffle(client_indices[cid])
    return client_indices


def make_loader(x, y, indices, batch_size=32, shuffle=True):
    ds = TensorDataset(x[indices], y[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def weighted_average_state_dict(state_dicts, weights):
    total = float(sum(weights))
    out = OrderedDict()
    for key in state_dicts[0].keys():
        acc = state_dicts[0][key] * (weights[0] / total)
        for i in range(1, len(state_dicts)):
            acc = acc + state_dicts[i][key] * (weights[i] / total)
        out[key] = acc
    return out


def extract_feature_batch(model, data, device):
    model = model.to(device)
    with torch.no_grad():
        x = model.backbone(data.to(device))
        x = x.view(x.size(0), -1)
        x = torch.relu(model.backbone_proj(x))
    return x


def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    return 100.0 * correct / max(total, 1)


def local_train_full(model, loader, device, local_epochs=1, lr=0.05):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.cpu().state_dict())


def local_train_fedmoe(model, global_backbone, loader, device, local_epochs=1, lr=0.05, lambda_consistency=0.01):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    global_ref = {k: v.to(device) for k, v in global_backbone.items()}

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            task_loss = criterion(logits, target)
            reg = torch.tensor(0.0, device=device)
            state = model.state_dict()
            for key in global_backbone.keys():
                reg = reg + nn.functional.mse_loss(state[key], global_ref[key])
            loss = task_loss + lambda_consistency * reg
            loss.backward()
            optimizer.step()

    out_state = model.cpu().state_dict()
    out_backbone = OrderedDict(
        (k, v) for k, v in out_state.items() if k.startswith("backbone") or k.startswith("backbone_proj")
    )
    return copy.deepcopy(out_state), out_backbone


def sync_global_backbone_to_clients(client_models, global_backbone):
    for model in client_models:
        state = model.state_dict()
        for key, value in global_backbone.items():
            state[key] = value
        model.load_state_dict(state)


def build_router_feature_dataset(client_models, global_backbone, client_loaders, device, max_samples_per_client=256):
    sync_global_backbone_to_clients(client_models, global_backbone)
    feats_all, labels_all = [], []
    for cid, (model, loader) in enumerate(zip(client_models, client_loaders)):
        collected = 0
        for data, _ in loader:
            feat = extract_feature_batch(model, data, device).cpu()
            feats_all.append(feat)
            labels_all.append(torch.full((feat.size(0),), cid, dtype=torch.long))
            collected += feat.size(0)
            if collected >= max_samples_per_client:
                break
    return TensorDataset(torch.cat(feats_all, dim=0), torch.cat(labels_all, dim=0))


def train_router(
    router,
    client_models,
    global_backbone,
    client_train_loaders,
    device,
    epochs=5,
    lr=1e-2,
    max_samples_per_client=256,
):
    ds = build_router_feature_dataset(
        client_models, global_backbone, client_train_loaders, device, max_samples_per_client=max_samples_per_client
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    router = router.to(device)
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(router(x), y)
            loss.backward()
            opt.step()


def evaluate_router_top1(router, client_models, global_backbone, client_test_loaders, device):
    ds = build_router_feature_dataset(
        client_models, global_backbone, client_test_loaders, device, max_samples_per_client=256
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    router = router.to(device)
    router.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = router(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(total, 1)


def evaluate_personalized_fedavg(global_model, client_test_loaders, device):
    return float(np.mean([evaluate_model(global_model, loader, device) for loader in client_test_loaders]))


def evaluate_personalized_fedmoe_oracle(client_models, global_backbone, client_test_loaders, device):
    sync_global_backbone_to_clients(client_models, global_backbone)
    return float(np.mean([evaluate_model(model, loader, device) for model, loader in zip(client_models, client_test_loaders)]))


def evaluate_personalized_fedmoe_router(client_models, global_backbone, router, client_test_loaders, device):
    sync_global_backbone_to_clients(client_models, global_backbone)
    models = [m.to(device).eval() for m in client_models]
    backbone_ref = models[0]
    router = router.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for loader in client_test_loaders:
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                feats = extract_feature_batch(backbone_ref, data, device)
                routed_idx = router(feats).argmax(dim=1)
                logits = []
                for i in range(data.size(0)):
                    expert = models[routed_idx[i].item()]
                    logits.append(expert.expert(feats[i : i + 1]))
                pred = torch.cat(logits, dim=0).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
    return 100.0 * correct / max(total, 1)


def run_benchmark(args):
    setup_seed(args.seed)
    device = torch.device("cpu")
    x_train, y_train, x_test, y_test, num_classes = load_dataset(args.dataset_name, seed=args.seed)

    train_part = build_noniid_partition(y_train, args.num_clients, args.labels_per_client, num_classes=num_classes)
    test_part = build_noniid_partition(y_test, args.num_clients, args.labels_per_client, num_classes=num_classes)

    train_loaders = [make_loader(x_train, y_train, train_part[cid], batch_size=args.batch_size) for cid in range(args.num_clients)]
    test_loaders = [make_loader(x_test, y_test, test_part[cid], batch_size=args.batch_size, shuffle=False) for cid in range(args.num_clients)]

    fedmoe_template = build_model(args.model_arch, num_classes=num_classes, hidden_dim=args.hidden_dim)
    fedmoe_backbone_params, fedmoe_expert_params = count_backbone_expert_params(fedmoe_template)
    fedmoe_unique_params = fedmoe_backbone_params + args.num_clients * fedmoe_expert_params

    if args.fedavg_hidden_dim > 0:
        fedavg_hidden_dim = args.fedavg_hidden_dim
    elif args.param_budget == "match_total":
        fedavg_hidden_dim = find_hidden_dim_for_target(args.model_arch, num_classes, fedmoe_unique_params)
    else:
        # same_hidden: compare under same base width (active-path fairness).
        fedavg_hidden_dim = args.hidden_dim

    fedavg_global = build_model(args.model_arch, num_classes=num_classes, hidden_dim=fedavg_hidden_dim)
    fedavg_params = count_model_params(fedavg_global)
    fedavg_flops = estimate_flops_per_sample(
        args.model_arch, fedavg_hidden_dim, num_classes, args.num_clients, include_router=False
    )

    fedmoe_clients = [build_model(args.model_arch, num_classes=num_classes, hidden_dim=args.hidden_dim) for _ in range(args.num_clients)]
    fedmoe_global_backbone = OrderedDict(
        (k, v) for k, v in fedmoe_clients[0].state_dict().items() if k.startswith("backbone") or k.startswith("backbone_proj")
    )
    router = RouterLinear(in_dim=args.hidden_dim, out_dim=args.num_clients)
    fedmoe_oracle_flops = estimate_flops_per_sample(
        args.model_arch, args.hidden_dim, num_classes, args.num_clients, include_router=False
    )
    fedmoe_router_flops = estimate_flops_per_sample(
        args.model_arch, args.hidden_dim, num_classes, args.num_clients, include_router=True
    )

    k = max(1, int(args.frac * args.num_clients))
    hist = {"fedavg": [], "fedmoe_oracle": [], "fedmoe_router": [], "router_top1": []}

    for _ in range(args.rounds):
        selected = random.sample(list(range(args.num_clients)), k)

        # FedAvg
        local_states, local_weights = [], []
        for cid in selected:
            local_model = copy.deepcopy(fedavg_global)
            local_states.append(local_train_full(local_model, train_loaders[cid], device, args.local_epochs, args.lr))
            local_weights.append(len(train_part[cid]))
        fedavg_global.load_state_dict(weighted_average_state_dict(local_states, local_weights))

        # FedMoE
        local_backbones, local_weights = [], []
        for cid in selected:
            state = fedmoe_clients[cid].state_dict()
            for key, value in fedmoe_global_backbone.items():
                state[key] = value
            fedmoe_clients[cid].load_state_dict(state)
            new_state, new_backbone = local_train_fedmoe(
                fedmoe_clients[cid],
                fedmoe_global_backbone,
                train_loaders[cid],
                device,
                args.local_epochs,
                args.lr,
                args.lambda_consistency,
            )
            fedmoe_clients[cid].load_state_dict(new_state)
            local_backbones.append(new_backbone)
            local_weights.append(len(train_part[cid]))
        fedmoe_global_backbone = weighted_average_state_dict(local_backbones, local_weights)

        train_router(
            router,
            fedmoe_clients,
            fedmoe_global_backbone,
            train_loaders,
            device,
            epochs=args.router_epochs,
            lr=args.router_lr,
            max_samples_per_client=args.router_max_samples,
        )

        hist["fedavg"].append(evaluate_personalized_fedavg(fedavg_global, test_loaders, device))
        hist["fedmoe_oracle"].append(evaluate_personalized_fedmoe_oracle(fedmoe_clients, fedmoe_global_backbone, test_loaders, device))
        hist["fedmoe_router"].append(
            evaluate_personalized_fedmoe_router(fedmoe_clients, fedmoe_global_backbone, router, test_loaders, device)
        )
        hist["router_top1"].append(evaluate_router_top1(router, fedmoe_clients, fedmoe_global_backbone, test_loaders, device))

    return {
        "fedavg_final": hist["fedavg"][-1],
        "fedmoe_oracle_final": hist["fedmoe_oracle"][-1],
        "fedmoe_router_final": hist["fedmoe_router"][-1],
        "router_top1_final": hist["router_top1"][-1],
        "routing_gap_final": hist["fedmoe_oracle"][-1] - hist["fedmoe_router"][-1],
        "gain_router_final": hist["fedmoe_router"][-1] - hist["fedavg"][-1],
        "fedavg_hidden_dim": fedavg_hidden_dim,
        "fedmoe_hidden_dim": args.hidden_dim,
        "fedavg_params": fedavg_params,
        "fedmoe_unique_params": int(fedmoe_unique_params),
        "fedmoe_backbone_params": int(fedmoe_backbone_params),
        "fedmoe_expert_params": int(fedmoe_expert_params),
        "fedavg_flops_per_sample": fedavg_flops,
        "fedmoe_oracle_flops_per_sample": fedmoe_oracle_flops,
        "fedmoe_router_flops_per_sample": fedmoe_router_flops,
        "param_budget_mode": args.param_budget,
        "history": hist,
    }


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark: FedAvg vs FedMoE (oracle/router)")
    parser.add_argument("--dataset_name", type=str, default="digits", choices=["digits", "wine", "iris", "breast_cancer"])
    parser.add_argument("--model_arch", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--labels_per_client", type=int, default=2)
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
    args = parser.parse_args()

    res = run_benchmark(args)
    print("==== CPU Benchmark ({}, {}) ====".format(args.dataset_name, args.model_arch))
    print(
        "Param Budget [{}] | FedAvg hidden_dim {} | FedMoE hidden_dim {}".format(
            res["param_budget_mode"], res["fedavg_hidden_dim"], res["fedmoe_hidden_dim"]
        )
    )
    print(
        "Params | FedAvg {:,} | FedMoE unique {:,} (backbone {:,} + {}xexpert {:,})".format(
            res["fedavg_params"],
            res["fedmoe_unique_params"],
            res["fedmoe_backbone_params"],
            args.num_clients,
            res["fedmoe_expert_params"],
        )
    )
    print(
        "FLOPs/sample | FedAvg {:,} | FedMoE-Oracle {:,} | FedMoE-Router {:,}".format(
            res["fedavg_flops_per_sample"],
            res["fedmoe_oracle_flops_per_sample"],
            res["fedmoe_router_flops_per_sample"],
        )
    )
    print(
        "Final | FedAvg {:.2f}% | FedMoE-Oracle {:.2f}% | FedMoE-Router {:.2f}%".format(
            res["fedavg_final"], res["fedmoe_oracle_final"], res["fedmoe_router_final"]
        )
    )
    print(
        "Router Top1 {:.2f}% | Routing Gap {:.2f}% | Router Gain {:+.2f}%".format(
            res["router_top1_final"], res["routing_gap_final"], res["gain_router_final"]
        )
    )


if __name__ == "__main__":
    main()
