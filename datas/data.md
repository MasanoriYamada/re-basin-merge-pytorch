# mnist-fmnist
## mnsit
augmax.ByteToFloat(): 0-255 -> 0-1: ToTensor()

## fmnist
same as mnist


# upsp-mnist
## usps, mnist
transform = transforms.Compose([transforms.Resize([32, 32]),
                                          transforms.Grayscale(3),
                                          transforms.ToTensor()])

## cifar10
### git-rebasin
train
augmax.RandomSizedCrop(32, 32, zoom_range=(0.8, 1.2)): RandomResizedCrop(32, scale=(0.8, 1.2))
augmax.HorizontalFlip(): RandomHorizontalFlip()
augmax.Rotate(): RandomRotation(degrees=(-30,30))
augmax.ByteToFloat(): 0-255 -> 0-1: ToTensor()
augmax.Normalize(): (x - mean) / std: Normalize()

> mean: [0.485, 0.456, 0.406] 
> std: [0.229, 0.224, 0.225]

### repair
https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-Merge-REPAIR-VGG11.ipynb
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    normalize,
])
test_transform = T.Compose([
    T.ToTensor(),
    normalize,
])


# Ref
- augmax: https://github.com/khdlr/augmax/blob/master/augmax/geometric.py
- torchvision: https://pytorch.org/vision/stable/transforms.html