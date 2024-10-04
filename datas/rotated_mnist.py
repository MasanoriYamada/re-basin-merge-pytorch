import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate


angles = [0, 15, 30, 45, 60, 75, 90]


def get_train_data(angle, trans, length):
    if trans is None:
        trans = transforms.Compose([
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])
    else:
        trans = transforms.Compose([
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            *trans.transforms,
            transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=trans)
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_test_data(angle, trans, length):
    if trans is None:
        trans = transforms.Compose([
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])
    else:
        trans = transforms.Compose([
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            *trans.transforms,
            transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=trans)
    if length is not None:
        dataset.data = dataset.data[:length]
        dataset.targets = dataset.targets[:length]
    return dataset


def get_train_loader(angle, batch_size, trans=None, length=None):
    dataset = get_train_data(angle, trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


def get_test_loader(angle, batch_size, trans=None, length=None):
    dataset = get_test_data(angle, trans, length)
    kwargs = {'batch_size': batch_size, 'shuffle': False}
    return torch.utils.data.DataLoader(dataset=dataset, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    loader = get_train_loader(angle=45, batch_size=64, length=140)
    for x, t in loader:
        print(x.shape, t.shape)
