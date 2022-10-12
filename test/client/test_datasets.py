# coding: utf-8
import torch

from flearn.client.datasets.get_data import DictDataset, RandomDataset, get_dataloader

if __name__ == "__main__":
    random_dataset = RandomDataset(200)
    dict_dataset = DictDataset(
        {
            1: torch.rand(10, 3, 32, 32),
            2: torch.rand(10, 3, 32, 32),
        }
    )

    print(len(random_dataset), len(dict_dataset))
    trainloader, testloader = get_dataloader(random_dataset, dict_dataset, batch_size=64)
