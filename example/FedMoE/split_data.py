# coding: utf-8
import numpy as np


def _get_targets(dataset):
    targets = dataset.targets
    if hasattr(targets, "tolist"):
        return targets.tolist()
    return list(targets)


def fedmoe_noniid(dataset, num_users, num_experts):
    """Split dataset into class-specialized client shards for FedMoE."""
    targets = _get_targets(dataset)
    num_classes = len(set(targets))

    labels_per_expert = [[] for _ in range(num_experts)]
    for label in range(num_classes):
        labels_per_expert[label % num_experts].append(label)

    expert_to_clients = {eid: [] for eid in range(num_experts)}
    for cid in range(num_users):
        expert_to_clients[cid % num_experts].append(cid)

    label_to_indices = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(targets):
        label_to_indices[label].append(idx)
    for label in range(num_classes):
        np.random.shuffle(label_to_indices[label])

    client_indices = {cid: [] for cid in range(num_users)}
    for expert_id, labels in enumerate(labels_per_expert):
        clients = expert_to_clients[expert_id]
        if len(clients) == 0:
            continue
        for label in labels:
            parts = np.array_split(label_to_indices[label], len(clients))
            for cid, part in zip(clients, parts):
                if len(part) > 0:
                    client_indices[cid].extend(part.tolist())

    for cid in range(num_users):
        np.random.shuffle(client_indices[cid])

    client_expert_map = {cid: cid % num_experts for cid in range(num_users)}
    return client_indices, client_expert_map
