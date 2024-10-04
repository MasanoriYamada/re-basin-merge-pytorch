import torch

from datas.mnist import get_train_data as mnist_get_train_data
from datas.mnist import get_test_data as mnist_get_test_data
from datas.fmnist import get_train_data as fmnist_get_train_data
from datas.fmnist import get_test_data as fmnist_get_test_data


def get_train_data(trans_a, trans_b, length):
    m_dataset = mnist_get_train_data(trans_a, length)
    fm_dataset = fmnist_get_train_data(trans_b, length)
    dataset = torch.utils.data.ConcatDataset([m_dataset, fm_dataset])
    return dataset


def get_test_data(trans_a, trans_b, length):
    m_dataset = mnist_get_test_data(trans_a, length)
    fm_dataset = fmnist_get_test_data(trans_b, length)
    dataset = torch.utils.data.ConcatDataset([m_dataset, fm_dataset])
    return dataset


def get_train_loader(batch_size, trans_mnist=None, trans_fmnist=None, length=None):
    dataset = get_train_data(trans_mnist, trans_fmnist, length)
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


def get_test_loader(batch_size, trans_mnist=None, trans_fmnist=None, length=None):
    dataset = get_test_data(trans_mnist, trans_fmnist, length)
    kwargs = {'batch_size': batch_size, 'shuffle': False}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    loader = get_train_loader(batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
