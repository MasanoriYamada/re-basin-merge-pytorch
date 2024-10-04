import torch
from torch.utils.data.sampler import WeightedRandomSampler


def get_train_loader(batch_size, dataset_a, dataset_b):
    a_size = len(dataset_a)
    b_size = len(dataset_b)
    dataset = torch.utils.data.ConcatDataset([dataset_a, dataset_b])
    dataset.classes = None  # can not use clip
    if a_size > b_size:
        a_weights = [1] * a_size
        b_weights = [a_size / b_size] * b_size
        weights = a_weights + b_weights
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
        kwargs = {'batch_size': batch_size, 'sampler': sampler}
    elif a_size < b_size:
        a_weights = [b_size / a_size] * a_size
        b_weights = [1] * b_size
        weights = a_weights + b_weights
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
        kwargs = {'batch_size': batch_size, 'sampler': sampler}
    elif a_size == b_size:
        sampler = None
        kwargs = {'batch_size': batch_size, 'shuffle': True}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


def get_test_loader(batch_size, dataset_a, dataset_b):
    a_size = len(dataset_a)
    b_size = len(dataset_b)
    if a_size != b_size:
        print('Warning: Because the accuracy is biased by the different sizes of the two data sets.')
    dataset = torch.utils.data.ConcatDataset([dataset_a, dataset_b])
    dataset.classes = None  # can not use clip
    kwargs = {'batch_size': batch_size, 'shuffle': False}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    loader = get_train_loader(batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
