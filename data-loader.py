# laoding Fashion-MNIST from TorchVision
    # `root` is the path where the train/test data is stored 
    # `train` specifies training or test dataset
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Store datasets in the project's data folder, regardless of the working directory.
DATA_DIR = Path(__file__).resolve().parent / "data"

FashionMNIST_training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # this composition turns image->ToImage()->ToDtype(torch.float32, scale=True)
)
FashionMNIST_test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # this composition turns image->ToImage()->ToDtype(torch.float32, scale=True)
)

MNIST_training_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
MNIST_test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) 
)


CIFAR10_training_data = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
CIFAR10_test_data = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) 
)

# DTD images vary in size; use the same fixed dimensions for train and test.
dtd_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

DTD_training_data = datasets.DTD(
    root=DATA_DIR,
    split="train",
    download=True,
    transform=dtd_transform,
)
DTD_test_data = datasets.DTD(
    root=DATA_DIR,
    split="test",
    download=True,
    transform=dtd_transform,
)



