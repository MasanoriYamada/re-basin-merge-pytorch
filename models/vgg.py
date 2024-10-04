from torch import nn
# https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-and-Permute-CIFAR10-VGG11-BatchNorm.ipynb

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, w=1, input_shape=3*32*32):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.w = w
        self.features = self._make_layers(cfg[vgg_name])
        if input_shape == 3*32*32:
            self.classifier = nn.Linear(self.w*512, num_classes)
        elif input_shape == 3*96*96:
            self.classifier = nn.Linear(self.w*4608, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return nn.functional.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(self.w*x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    import torch
    from torchsummary import summary

    net = VGG('VGG11', 10)
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

    net = VGG('VGG16', 10)
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

    net = VGG('VGG11', 126, w=1, input_shape=3*96*96)
    print(net)
    x = torch.randn(2, 3, 96, 96)
    y = net(x)
    print(y.size())
    summary(net, input_size=(3, 96, 96))


if __name__ == "__main__":
    test()