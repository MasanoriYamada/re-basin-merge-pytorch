import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input=28 * 28, num_classes=10, bias=True):
        super().__init__()
        self.input = input
        self.layer0 = nn.Linear(input, 512, bias=bias)
        self.layer1 = nn.Linear(512, 512, bias=bias)
        self.layer2 = nn.Linear(512, 512, bias=bias)
        self.layer3 = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return nn.functional.log_softmax(x, dim=1)


class MLP2(nn.Module):
    # no use
    def __init__(self, input=28 * 28, num_classes=10, bias=True):
        super().__init__()
        self.input = input
        self.layer0 = nn.Linear(self.input, 512, bias=bias)
        self.layer1 = nn.Linear(512, 512, bias=bias)
        self.layer2 = nn.Linear(512, 512, bias=bias)
        self.layer3 = nn.Linear(512, 256, bias=bias)
        self.layer4 = nn.Linear(256, num_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x
