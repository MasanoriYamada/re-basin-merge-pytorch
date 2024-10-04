import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10, clip=False, classes=None, device='cuda'):
        super(ResNet, self).__init__()
        self.in_planes = w * 16
        self.clip = clip
        if clip:
            assert classes is not None
            self.class_vectors = load_clip_features(classes, device=device)
        self.conv1 = nn.Conv2d(3, w * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w * 16)
        self.layer1 = self._make_layer(block, w * 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w * 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w * 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(w * 64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.clip:
            normed_out = out / out.norm(dim=-1, keepdim=True)
            self.class_vectors = self.class_vectors.to(normed_out.device)
            self.class_vectors = self.class_vectors.to(normed_out.dtype)
            out = (100.0 * normed_out @ self.class_vectors.T)
        return nn.functional.log_softmax(out, dim=1)


def resnet20(w=1, num_classes=10, clip=False, classes=None, device='cuda'):
    return ResNet(BasicBlock, [3, 3, 3], w=w, num_classes=num_classes, clip=clip, classes=classes, device=device)


def load_clip_features(class_names, device):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    model, preprocess = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def test():
    import torch
    from torchsummary import summary
    net = resnet20(1, 10)
    print(net)
    # x = torch.randn(2, 3, 32, 32)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
    summary(net, input_size=(3, 224, 224))

    # net = resnet20(4, 126)
    # x = torch.randn(2, 3, 96, 96)
    # y = net(x)
    # print(y.size())
    # summary(net, input_size=(3, 96, 96))


if __name__ == "__main__":
    test()
