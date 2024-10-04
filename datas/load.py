import os
import torch
import numpy as np
from collections.abc import Iterable
from torchvision import transforms
from torchvision.transforms import functional
from torch.nn.functional import interpolate


class Table(object):
    def __init__(self, paths: list, transform=None):
        self.transform = transform
        xs = []
        ts = []
        for path in paths:
            ext = os.path.splitext(path)
            if ext[1] == '.pt':
                print(f'load: {path}')
                dat = torch.load(path)
                xs.append(dat["data"][0][0])
                ts.append(dat["data"][0][1])
            elif ext[1] =='.npz':
                print(f'load: {path}')
                dat = np.load(path)
                x = torch.from_numpy(dat['images']).permute(0, 3, 1, 2)  # b,h,w,c -> b,c,h,w
                # normalize
                x = functional.normalize(x, x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3)))
                xs.append(x)
                ts.append(torch.from_numpy(dat['labels']).argmax(axis=1))
        if len(xs) == 2:
            if xs[0].shape != xs[1].shape:
                # check transforms include Resize and Grayscale
                has_resize = any(isinstance(t, transforms.Resize) for t in self.transform.transforms)
                has_grayscale = any(isinstance(t, transforms.Grayscale) for t in self.transform.transforms)
                assert has_resize, "Resize transform is missing in the transform pipeline."
                assert has_grayscale, "Grayscale transform is missing in the transform pipeline."
                def resize_tensor(tensor, size):
                    return interpolate(tensor, size=size, mode='bilinear', align_corners=False)
                # 3x32x32 for mnist, usps
                x1 = resize_tensor(xs[0], (32, 32)).repeat_interleave(3, dim=1)
                x2 = resize_tensor(xs[1], (32, 32)).repeat_interleave(3, dim=1)
                xs = [x1, x2]
        self.data = torch.cat(xs)
        self.label = torch.cat(ts)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        modified_transform = replace_to_tensor(self.transform)
        img = modified_transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class ToTensorIfNeeded:
    """
    If the input is a PIL Image or ndarray, ToTensor is applied. If it is already a Tensor, return it as is.
    """
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return transforms.ToTensor()(pic)

def replace_to_tensor(transform_pipeline):
    """
    Replace ToTensor in the transforms.Compose object with ToTensorIfNeeded.
    """
    new_transforms = []
    for transform in transform_pipeline.transforms:
        if isinstance(transform, transforms.ToTensor):
            new_transforms.append(ToTensorIfNeeded())
        else:
            new_transforms.append(transform)
    return transforms.Compose(new_transforms)


class Loader(object):
    def __init__(self, paths: list, transform=None):
        self.paths = paths
        self.transform = transform  # train only

    def get_train_loader(self, batch_size):
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        train_dataset = Table(paths=self.paths, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, **train_kwargs)
        return train_loader

    def get_test_loader(self, batch_size):
        test_kwargs = {'batch_size': batch_size, 'shuffle': False}
        test_dataset = Table(paths=self.paths)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, **test_kwargs)
        return test_loader


if __name__ == "__main__":
    loader = Loader()
    loader.get_train_loader(batch_size=64)
