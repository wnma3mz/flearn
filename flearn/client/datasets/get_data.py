# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST

from .partition_data import DataPartitioner


def create_data_randomly(sample_number, pn_normalize=True):
    # create pseudo_data and map to [0, 1].
    # cifar10
    pseudo_data = torch.randn((sample_number, 3, 32, 32), requires_grad=False)
    pseudo_data = (pseudo_data - torch.min(pseudo_data)) / (torch.max(pseudo_data) - torch.min(pseudo_data))

    # map values to [-1, 1] if necessary.
    if pn_normalize:
        pseudo_data = (pseudo_data - 0.5) * 2
    return pseudo_data, [0] * sample_number


class RandomDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, sample_number):
        "Initialization"
        self.data, self.labels = create_data_randomly(sample_number)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        return self.data[index], self.labels[index]


class DictDataset(Dataset):
    def __init__(self, label_data_d):
        "Initialization"
        self.data, self.labels = [], []
        for label, data in label_data_d.items():
            self.data.append(data)
            self.labels.append(torch.tensor([label] * len(data)))

        self.data = torch.cat(self.data).type(torch.float32)
        self.labels = torch.cat(self.labels).type(torch.long)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        return self.data[index], self.labels[index]


class SingleClassDataset(Dataset):
    def __init__(self, dataset, label, transform=None, target_transform=None) -> None:
        """
        dataset = CIFAR10(
            self.root, self.train, self.transform, self.target_transform, self.download
        )
        """
        super().__init__()
        if type(dataset.targets) == list:
            dataset.targets = np.array(dataset.targets)
        self.data = dataset.data[dataset.targets == label]
        self.targets = dataset.targets[dataset.targets == label]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_datasets(dataset_name, dataset_dir, split=None):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trans_emnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trains_cifar10 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trains_cifar100 = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    trans_cifar_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            trains_cifar10,
        ]
    )
    # trans_cifar_train = transforms.Compose(
    #     [
    #         transforms.Resize([256, 256]),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation((-30, 30)),
    #         transforms.ToTensor(),
    #     ]
    # )

    trans_cifar_test = transforms.Compose([transforms.ToTensor(), trains_cifar10])

    if dataset_name == "mnist":
        trainset = MNIST(dataset_dir, train=True, download=False, transform=trans_mnist)
        testset = MNIST(dataset_dir, train=False, download=False, transform=trans_mnist)
    elif dataset_name == "cifar10":
        trainset = CIFAR10(dataset_dir, train=True, download=False, transform=trans_cifar_train)
        testset = CIFAR10(dataset_dir, train=False, download=False, transform=trans_cifar_test)
    elif dataset_name == "cifar100":
        trainset = CIFAR100(dataset_dir, train=True, download=False, transform=trans_cifar_train)
        testset = CIFAR100(dataset_dir, train=False, download=False, transform=trans_cifar_test)
    elif dataset_name == "emnist":
        trainset = EMNIST(dataset_dir, train=True, download=False, split=split, transform=trans_emnist)
        testset = EMNIST(
            dataset_dir,
            train=False,
            download=False,
            split=split,
            transform=trans_emnist,
        )
    else:
        raise NotImplementedError("This dataset is not currently supported")
    return trainset, testset


def get_dataloader(trainset, testset, batch_size, num_workers=0, pin_memory=False):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def get_split_loader(
    trainset,
    testset,
    trainloader_id,
    testloader_id,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    local_trainset = DatasetSplit(trainset, trainloader_id)
    local_testset = DatasetSplit(testset, testloader_id)
    trainloader = DataLoader(
        local_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        local_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainloader, testloader


def define_dataset(conf, dataset_name):
    # prepare general train/test.
    conf.partitioned_by_user = False
    train_dataset, test_dataset = get_datasets(dataset_name, conf.data_dir)

    # create the validation from train.
    train_dataset, val_dataset, test_dataset = define_val_dataset(conf, train_dataset, test_dataset)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def define_val_dataset(conf, train_dataset, test_dataset):
    assert conf.val_data_ratio >= 0

    partition_sizes = [
        (1 - conf.val_data_ratio) * conf.train_data_ratio,
        (1 - conf.val_data_ratio) * (1 - conf.train_data_ratio),
        conf.val_data_ratio,
    ]

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )
    train_dataset = data_partitioner.use(0)

    # split for val data.
    if conf.val_data_ratio > 0:
        assert conf.partitioned_by_user is False

        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset


def define_data_loader(conf, dataset, localdata_id=None, is_train=True, shuffle=True, data_partitioner=None):
    # determine the data to load,
    # either the whole dataset, or a subset specified by partition_type.
    if is_train:
        world_size = conf.n_clients
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        assert localdata_id is not None

        if conf.partitioned_by_user:  # partitioned by "users".
            # in case our dataset is already partitioned by the client.
            # and here we need to load the dataset based on the client id.
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:  # (general) partitioned by "labels".
            # in case we have a global dataset and want to manually partition them.
            if data_partitioner is None:
                # update the data_partitioner.
                data_partitioner = DataPartitioner(conf, dataset, partition_sizes, partition_type=conf.partition_data)
            # note that the master node will not consume the training dataset.
            data_to_load = data_partitioner.use(localdata_id)
    else:
        if conf.partitioned_by_user:  # partitioned by "users".
            # in case our dataset is already partitioned by the client.
            # and here we need to load the dataset based on the client id.
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:
            data_to_load = dataset
    # use Dataloader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    return data_loader, data_partitioner


class CONF:
    def __init__(
        self,
        non_iid_alpha,
        n_clients,
        local_n_epochs,
        batch_size,
        data_dir,
        num_workers=0,
    ):
        self.non_iid_alpha = non_iid_alpha
        self.val_data_ratio = 0.1
        self.train_data_ratio = 1
        self.partitioned_by_user = False
        self.partition_data = "non_iid_dirichlet"
        self.manual_seed = 7
        self.random_state = np.random.RandomState(self.manual_seed)
        self.data_dir = data_dir
        self.n_clients = n_clients
        self.num_workers = num_workers
        self.pin_memory = True
        self.batch_size = batch_size
        self.local_n_epochs = local_n_epochs


if __name__ == "__main__":
    # 这里的评估是全局的test_loader，用val_loader进行drop worst
    conf = CONF(non_iid_alpha=1, n_clients=20, local_n_epochs=40, batch_size=64)
    torch.manual_seed(conf.manual_seed)

    dataset = define_dataset(conf, dataset_name="cifar100", display_log=True)
    _, data_partitioner = define_data_loader(
        conf,
        dataset=dataset["train"],
        localdata_id=0,  # random id here.
        is_train=True,
        data_partitioner=None,
    )

    client_id = 0
    train_loader, _ = define_data_loader(
        conf,
        dataset=dataset["train"],
        localdata_id=client_id,
        is_train=True,
        data_partitioner=data_partitioner,
    )

    val_loader, _ = define_data_loader(conf, dataset["val"], is_train=False)
    test_loader, _ = define_data_loader(conf, dataset["test"], is_train=False)
    test_loaders = [test_loader]

    nn.CrossEntropyLoss(reduction="mean")
