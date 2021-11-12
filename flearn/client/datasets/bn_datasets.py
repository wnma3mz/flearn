import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# https://github.com/med-air/FedBN/blob/df4a9f9c4f/utils/data_utils.py
class DigitsDataset(Dataset):
    def __init__(
        self,
        data_path,
        channels,
        percent=0.1,
        filename=None,
        train=True,
        transform=None,
    ):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent * 10)):
                        if part == 0:
                            self.images, self.labels = np.load(
                                os.path.join(
                                    data_path,
                                    "partitions/train_part{}.pkl".format(part),
                                ),
                                allow_pickle=True,
                            )
                        else:
                            images, labels = np.load(
                                os.path.join(
                                    data_path,
                                    "partitions/train_part{}.pkl".format(part),
                                ),
                                allow_pickle=True,
                            )
                            self.images = np.concatenate([self.images, images], axis=0)
                            self.labels = np.concatenate([self.labels, labels], axis=0)
                else:
                    self.images, self.labels = np.load(
                        os.path.join(data_path, "partitions/train_part0.pkl"),
                        allow_pickle=True,
                    )
                    data_len = int(self.images.shape[0] * percent * 10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(
                    os.path.join(data_path, "test.pkl"), allow_pickle=True
                )
        else:
            self.images, self.labels = np.load(
                os.path.join(data_path, filename), allow_pickle=True
            )

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode="L")
        elif self.channels == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(
                os.path.join(
                    base_path, "office_caltech_10", "{}_train.pkl".format(site)
                ),
                allow_pickle=True,
            )
        else:
            self.paths, self.text_labels = np.load(
                os.path.join(
                    base_path, "office_caltech_10", "{}_test.pkl".format(site)
                ),
                allow_pickle=True,
            )

        label_dict = {
            "back_pack": 0,
            "bike": 1,
            "calculator": 2,
            "headphones": 3,
            "keyboard": 4,
            "laptop_computer": 5,
            "monitor": 6,
            "mouse": 7,
            "mug": 8,
            "projector": 9,
        }
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else "../data"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(
                os.path.join(base_path, "DomainNet", "{}_train.pkl".format(site)),
                allow_pickle=True,
            )
        else:
            self.paths, self.text_labels = np.load(
                os.path.join(base_path, "DomainNet", "{}_test.pkl".format(site)),
                allow_pickle=True,
            )

        label_dict = {
            "bird": 0,
            "feather": 1,
            "headphones": 2,
            "ice_cream": 3,
            "teapot": 4,
            "tiger": 5,
            "whale": 6,
            "windmill": 7,
            "wine_glass": 8,
            "zebra": 9,
        }

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else "../data"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
def prepare_digits_data(fpath, percent=0.1, batch=32, num_workers=0):
    # Prepare data
    transform_mnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_svhn = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_usps = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_synth = transforms.Compose(
        [
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_mnistm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # MNIST
    mnist_trainset = DigitsDataset(
        data_path=os.path.join(fpath, "MNIST"),
        channels=1,
        percent=percent,
        train=True,
        transform=transform_mnist,
    )
    mnist_testset = DigitsDataset(
        data_path=os.path.join(fpath, "MNIST"),
        channels=1,
        percent=percent,
        train=False,
        transform=transform_mnist,
    )

    # SVHN
    svhn_trainset = DigitsDataset(
        data_path=os.path.join(fpath, "SVHN"),
        channels=3,
        percent=percent,
        train=True,
        transform=transform_svhn,
    )
    svhn_testset = DigitsDataset(
        data_path=os.path.join(fpath, "SVHN"),
        channels=3,
        percent=percent,
        train=False,
        transform=transform_svhn,
    )

    # USPS
    usps_trainset = DigitsDataset(
        data_path=os.path.join(fpath, "USPS"),
        channels=1,
        percent=percent,
        train=True,
        transform=transform_usps,
    )
    usps_testset = DigitsDataset(
        data_path=os.path.join(fpath, "USPS"),
        channels=1,
        percent=percent,
        train=False,
        transform=transform_usps,
    )

    # Synth Digits
    synth_trainset = DigitsDataset(
        data_path=os.path.join(fpath, "SynthDigits"),
        channels=3,
        percent=percent,
        train=True,
        transform=transform_synth,
    )
    synth_testset = DigitsDataset(
        data_path=os.path.join(fpath, "SynthDigits"),
        channels=3,
        percent=percent,
        train=False,
        transform=transform_synth,
    )

    # MNIST-M
    mnistm_trainset = DigitsDataset(
        data_path=os.path.join(fpath, "MNIST_M"),
        channels=3,
        percent=percent,
        train=True,
        transform=transform_mnistm,
    )
    mnistm_testset = DigitsDataset(
        data_path=os.path.join(fpath, "MNIST_M"),
        channels=3,
        percent=percent,
        train=False,
        transform=transform_mnistm,
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        mnist_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    svhn_train_loader = torch.utils.data.DataLoader(
        svhn_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    svhn_test_loader = torch.utils.data.DataLoader(
        svhn_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    usps_train_loader = torch.utils.data.DataLoader(
        usps_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    usps_test_loader = torch.utils.data.DataLoader(
        usps_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    synth_train_loader = torch.utils.data.DataLoader(
        synth_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    synth_test_loader = torch.utils.data.DataLoader(
        synth_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    mnistm_train_loader = torch.utils.data.DataLoader(
        mnistm_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    mnistm_test_loader = torch.utils.data.DataLoader(
        mnistm_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    train_loaders = [
        mnist_train_loader,
        svhn_train_loader,
        usps_train_loader,
        synth_train_loader,
        mnistm_train_loader,
    ]
    test_loaders = [
        mnist_test_loader,
        svhn_test_loader,
        usps_test_loader,
        synth_test_loader,
        mnistm_test_loader,
    ]

    return train_loaders, test_loaders


# https://github.com/med-air/FedBN/blob/master/federated/fed_domainnet.py
def prepare_domainnet_data(fpath, batch=32, num_workers=0):
    transform_train = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    # clipart
    clipart_trainset = DomainNetDataset(fpath, "clipart", transform=transform_train)
    clipart_testset = DomainNetDataset(
        fpath, "clipart", transform=transform_test, train=False
    )
    # infograph
    infograph_trainset = DomainNetDataset(fpath, "infograph", transform=transform_train)
    infograph_testset = DomainNetDataset(
        fpath, "infograph", transform=transform_test, train=False
    )
    # painting
    painting_trainset = DomainNetDataset(fpath, "painting", transform=transform_train)
    painting_testset = DomainNetDataset(
        fpath, "painting", transform=transform_test, train=False
    )
    # quickdraw
    quickdraw_trainset = DomainNetDataset(fpath, "quickdraw", transform=transform_train)
    quickdraw_testset = DomainNetDataset(
        fpath, "quickdraw", transform=transform_test, train=False
    )
    # real
    real_trainset = DomainNetDataset(fpath, "real", transform=transform_train)
    real_testset = DomainNetDataset(
        fpath, "real", transform=transform_test, train=False
    )
    # sketch
    sketch_trainset = DomainNetDataset(fpath, "sketch", transform=transform_train)
    sketch_testset = DomainNetDataset(
        fpath, "sketch", transform=transform_test, train=False
    )

    min_data_len = min(
        len(clipart_trainset),
        len(infograph_trainset),
        len(painting_trainset),
        len(quickdraw_trainset),
        len(real_trainset),
        len(sketch_trainset),
    )
    val_len = int(min_data_len * 0.05)
    min_data_len = int(min_data_len * 0.05)

    clipart_valset = torch.utils.data.Subset(
        clipart_trainset, list(range(len(clipart_trainset)))[-val_len:]
    )
    clipart_trainset = torch.utils.data.Subset(
        clipart_trainset, list(range(min_data_len))
    )

    infograph_valset = torch.utils.data.Subset(
        infograph_trainset, list(range(len(infograph_trainset)))[-val_len:]
    )
    infograph_trainset = torch.utils.data.Subset(
        infograph_trainset, list(range(min_data_len))
    )

    painting_valset = torch.utils.data.Subset(
        painting_trainset, list(range(len(painting_trainset)))[-val_len:]
    )
    painting_trainset = torch.utils.data.Subset(
        painting_trainset, list(range(min_data_len))
    )

    quickdraw_valset = torch.utils.data.Subset(
        quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:]
    )
    quickdraw_trainset = torch.utils.data.Subset(
        quickdraw_trainset, list(range(min_data_len))
    )

    real_valset = torch.utils.data.Subset(
        real_trainset, list(range(len(real_trainset)))[-val_len:]
    )
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    sketch_valset = torch.utils.data.Subset(
        sketch_trainset, list(range(len(sketch_trainset)))[-val_len:]
    )
    sketch_trainset = torch.utils.data.Subset(
        sketch_trainset, list(range(min_data_len))
    )

    clipart_train_loader = torch.utils.data.DataLoader(
        clipart_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    clipart_val_loader = torch.utils.data.DataLoader(
        clipart_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    clipart_test_loader = torch.utils.data.DataLoader(
        clipart_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    infograph_train_loader = torch.utils.data.DataLoader(
        infograph_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    infograph_val_loader = torch.utils.data.DataLoader(
        infograph_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    infograph_test_loader = torch.utils.data.DataLoader(
        infograph_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    painting_train_loader = torch.utils.data.DataLoader(
        painting_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    painting_val_loader = torch.utils.data.DataLoader(
        painting_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    painting_test_loader = torch.utils.data.DataLoader(
        painting_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    quickdraw_train_loader = torch.utils.data.DataLoader(
        quickdraw_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    quickdraw_val_loader = torch.utils.data.DataLoader(
        quickdraw_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    quickdraw_test_loader = torch.utils.data.DataLoader(
        quickdraw_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    real_train_loader = torch.utils.data.DataLoader(
        real_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    real_val_loader = torch.utils.data.DataLoader(
        real_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    real_test_loader = torch.utils.data.DataLoader(
        real_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    sketch_train_loader = torch.utils.data.DataLoader(
        sketch_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    sketch_val_loader = torch.utils.data.DataLoader(
        sketch_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    sketch_test_loader = torch.utils.data.DataLoader(
        sketch_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    train_loaders = [
        clipart_train_loader,
        infograph_train_loader,
        painting_train_loader,
        quickdraw_train_loader,
        real_train_loader,
        sketch_train_loader,
    ]
    val_loaders = [
        clipart_val_loader,
        infograph_val_loader,
        painting_val_loader,
        quickdraw_val_loader,
        real_val_loader,
        sketch_val_loader,
    ]
    test_loaders = [
        clipart_test_loader,
        infograph_test_loader,
        painting_test_loader,
        quickdraw_test_loader,
        real_test_loader,
        sketch_test_loader,
    ]

    return train_loaders, val_loaders, test_loaders


# https://github.com/med-air/FedBN/blob/master/federated/fed_office.py
def prepare_office_data(fpath, batch=32, num_workers=0):
    transform_office = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    # amazon
    amazon_trainset = OfficeDataset(fpath, "amazon", transform=transform_office)
    amazon_testset = OfficeDataset(
        fpath, "amazon", transform=transform_test, train=False
    )
    # caltech
    caltech_trainset = OfficeDataset(fpath, "caltech", transform=transform_office)
    caltech_testset = OfficeDataset(
        fpath, "caltech", transform=transform_test, train=False
    )
    # dslr
    dslr_trainset = OfficeDataset(fpath, "dslr", transform=transform_office)
    dslr_testset = OfficeDataset(fpath, "dslr", transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(fpath, "webcam", transform=transform_office)
    webcam_testset = OfficeDataset(
        fpath, "webcam", transform=transform_test, train=False
    )

    min_data_len = min(
        len(amazon_trainset),
        len(caltech_trainset),
        len(dslr_trainset),
        len(webcam_trainset),
    )
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(
        amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]
    )
    amazon_trainset = torch.utils.data.Subset(
        amazon_trainset, list(range(min_data_len))
    )

    caltech_valset = torch.utils.data.Subset(
        caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]
    )
    caltech_trainset = torch.utils.data.Subset(
        caltech_trainset, list(range(min_data_len))
    )

    dslr_valset = torch.utils.data.Subset(
        dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]
    )
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(
        webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]
    )
    webcam_trainset = torch.utils.data.Subset(
        webcam_trainset, list(range(min_data_len))
    )

    amazon_train_loader = torch.utils.data.DataLoader(
        amazon_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    amazon_val_loader = torch.utils.data.DataLoader(
        amazon_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    amazon_test_loader = torch.utils.data.DataLoader(
        amazon_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    caltech_train_loader = torch.utils.data.DataLoader(
        caltech_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    caltech_val_loader = torch.utils.data.DataLoader(
        caltech_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    caltech_test_loader = torch.utils.data.DataLoader(
        caltech_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    dslr_train_loader = torch.utils.data.DataLoader(
        dslr_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    dslr_val_loader = torch.utils.data.DataLoader(
        dslr_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    dslr_test_loader = torch.utils.data.DataLoader(
        dslr_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    webcam_train_loader = torch.utils.data.DataLoader(
        webcam_trainset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    webcam_val_loader = torch.utils.data.DataLoader(
        webcam_valset, batch_size=batch, shuffle=False, num_workers=num_workers
    )
    webcam_test_loader = torch.utils.data.DataLoader(
        webcam_testset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

    train_loaders = [
        amazon_train_loader,
        caltech_train_loader,
        dslr_train_loader,
        webcam_train_loader,
    ]
    val_loaders = [
        amazon_val_loader,
        caltech_val_loader,
        dslr_val_loader,
        webcam_val_loader,
    ]
    test_loaders = [
        amazon_test_loader,
        caltech_test_loader,
        dslr_test_loader,
        webcam_test_loader,
    ]
    return train_loaders, val_loaders, test_loaders
