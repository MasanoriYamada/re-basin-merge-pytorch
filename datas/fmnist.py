import torch
from torchvision import datasets, transforms


def get_train_data(trans, length):
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    dataset = datasets.FashionMNIST(root='./data/', train=True, download=True, transform=trans)
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_test_data(trans, length):
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    dataset = datasets.FashionMNIST(root='./data/', train=False, download=True, transform=trans)
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_train_loader(batch_size, trans=None, length=None):
    dataset = get_train_data(trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


def get_test_loader(batch_size, trans=None, length=None):
    dataset = get_test_data(trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': False}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    loader = get_train_loader(batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
