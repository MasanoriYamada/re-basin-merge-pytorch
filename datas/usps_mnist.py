import torch
from torch.utils.data.sampler import WeightedRandomSampler

from datas.mnist import get_train_data as mnist_get_train_data
from datas.mnist import get_test_data as mnist_get_test_data
from datas.usps import get_train_data as usps_get_train_data
from datas.usps import get_test_data as usps_get_test_data


def get_train_data(trans_a, trans_b, length):
    us_dataset = usps_get_train_data(trans_a, length)
    m_dataset = mnist_get_train_data(trans_b, length)
    usps_size = len(us_dataset)
    mnist_size = len(m_dataset)
    mnist_weights = [1] * mnist_size
    usps_weights = [mnist_size / usps_size] * usps_size
    weights = usps_weights + mnist_weights
    dataset = torch.utils.data.ConcatDataset([us_dataset, m_dataset])
    # Sampling to equal numbers due to different data sizes
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return dataset, sampler


def get_test_data(trans_a, trans_b, length):
    us_dataset = usps_get_test_data(trans_a, length)
    m_dataset = mnist_get_test_data(trans_b, length)
    dataset = torch.utils.data.ConcatDataset([us_dataset, m_dataset])
    return dataset


def get_train_loader(batch_size, trans_usps=None, trans_mnist=None, length=None):
    dataset, sampler = get_train_data(trans_usps, trans_mnist, length)
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    return torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, **kwargs)


def get_test_loader(batch_size, trans_usps=None, trans_mnist=None, length=None):
    print('Error: Because the accuracy is biased by the different sizes of the two data sets.')
    raise NotImplementedError()
    # dataset = get_test_data(trans_usps, trans_mnist, length)
    # kwargs = {'batch_size': batch_size, 'shuffle': False}
    # return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    loader = get_train_loader(batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
