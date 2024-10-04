import os
import numpy as np
import torch
from torchvision import datasets, transforms


def get_train_data(trans, length):
    # label 5 ~ 9: 25000
    split_label = 4
    if trans is None:
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=trans)
    np_target = np.array(dataset.targets)
    dataset.targets = np_target[np_target > split_label]
    dataset.targets = dataset.targets[dataset.targets > split_label]
    dataset.data = dataset.data[np_target > split_label]
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_test_data(trans, length):
    # label5 ~ 9: 5000
    split_label = 4
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    dataset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=trans)
    np_target = np.array(dataset.targets)
    dataset.targets = np_target[np_target > split_label]
    dataset.targets = dataset.targets[dataset.targets > split_label]
    dataset.data = dataset.data[np_target > split_label]
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_train_loader(batch_size, trans=None, length=None):
    dataset = get_train_data(trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': os.cpu_count()}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


def get_test_loader(batch_size, trans=None, length=None):
    dataset = get_test_data(trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': os.cpu_count()}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    loader = get_train_loader(batch_size=64, length=140)
    loader = get_train_loader(batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
