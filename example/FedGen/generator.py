import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAXLOG = 0.1

GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    "cifar10": (512, 64, 3, 10, 64),
    "cifar100": (512, 32, 3, 10, 64),
    # "cifar": (512, 84, 3, 10, 64),
    "celeb": (128, 32, 3, 2, 32),
    "mnist": (256, 32, 1, 10, 32),
    "mnist-cnn0": (256, 32, 1, 10, 64),
    "mnist-cnn1": (128, 32, 1, 10, 32),
    "mnist-cnn2": (64, 32, 1, 10, 32),
    "mnist-cnn3": (64, 32, 1, 10, 16),
    "emnist": (256, 32, 1, 25, 32),
    "emnist-cnn0": (256, 32, 1, 25, 64),
    "emnist-cnn1": (128, 32, 1, 25, 32),
    "emnist-cnn2": (128, 32, 1, 25, 16),
    "emnist-cnn3": (64, 32, 1, 25, 32),
}


class Generator(nn.Module):
    def __init__(
        self, dataset="mnist", model="cnn", embedding=False, latent_layer_idx=-1
    ):
        super(Generator, self).__init__()
        # print("Dataset {}".format(dataset))
        self.embedding = embedding
        self.dataset = dataset
        # self.model=model
        self.latent_layer_idx = latent_layer_idx
        (
            self.hidden_dim,
            self.latent_dim,
            self.input_channel,
            self.n_class,
            self.noise_dim,
        ) = GENERATORCONFIGS[dataset]
        input_dim = (
            self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        )
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)  # same as above
        self.diversity_loss = DiversityLoss(metric="l1")
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            # print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        # print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        device = labels.device

        result = {}
        batch_size = labels.shape[0]
        # sampling from Gaussian
        eps = torch.rand((batch_size, self.noise_dim), device=device)
        if verbose:
            result["eps"] = eps
        if self.embedding:  # embedded dense vector
            y_input = self.embedding_layer(labels)
        else:  # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            # labels = labels.view
            y_input.scatter_(1, labels.view(-1, 1), 1)
        y_input = y_input.to(device)
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result["output"] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = (
            layer.view((layer.size(0), layer.size(1), -1))
            .std(dim=2, keepdim=True)
            .unsqueeze(3)
        )
        return (layer - mean) / std


class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2 = torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2 = torch.split(layer, chunk_size, dim=0)
        lz = torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(torch.abs(eps1 - eps2))
        eps = 1 * 1e-5
        diversity_loss = 1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2 = torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2 = torch.split(layer, chunk_size, dim=0)
        lz = torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(torch.abs(eps1 - eps2))
        eps = 1 * 1e-5
        diversity_loss = 1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == "l1":
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == "l2":
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == "cosine":
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how="l2")
        return torch.exp(torch.mean(-noise_dist * layer_dist))
