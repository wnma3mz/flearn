# coding: utf-8
import base64
import os
import pickle
import sys

sys.path.append("../../..")
import torch
import torch.nn as nn
import torch.optim as optim
from flearn.client import Client
from flearn.client import Communicate as cc
from flearn.client.datasets import get_dataloader, get_datasets
from torch.utils.data import random_split
from torchvision.models import resnet18

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    pass
