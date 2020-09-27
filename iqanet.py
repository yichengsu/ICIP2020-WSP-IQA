import torch
import torch.nn as nn
import torchvision.models as models


class WSPBlock(nn.Module):
    def __init__(self, channels):
        super(WSPBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_weight = self.conv(x)
        x_weight = self.bn(x_weight)
        x_weight = self.relu(x_weight)
        out = torch.mul(x, x_weight).sum(dim=(2, 3)) / (x_weight.sum(dim=(2, 3)) + 0.0001)
        return out


class IQANet(nn.Module):
    def __init__(self):
        super(IQANet, self).__init__()
        self.resnet101_freeze = nn.Sequential(*list(models.resnet101(True).children())[:7])
        self.resnet101 = nn.Sequential(*list(models.resnet101(True).children())[7:8])
        self.wsp = WSPBlock(2048)
        self.fc = nn.Linear(2048, 5)

        # freeze conv and weight of batchnorm
        for para in self.resnet101_freeze.parameters():
            para.requires_grad = False

        # freeze running mean and var of barchnorm
        self.resnet101_freeze.eval()

    def forward(self, x):
        x = self.resnet101_freeze(x)
        x = self.resnet101(x)
        x = self.wsp(x)
        x = self.fc(x)
        return x

    def train(self, mode=True):
        self.training = mode

        for m in [self.resnet101, self.wsp, self.fc]:
            m.training = mode
            for module in m.children():
                module.train(mode)

        return self
