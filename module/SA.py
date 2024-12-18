#coding=utf-8
import torch
import torch.nn as nn
from module.weight_init import weight_init
from torch.nn.parameter import Parameter

class sa_layer(nn.Module):
    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out
    def initialize(self):
        weight_init(self)        
class WeightedGate(nn.Module):
    def __init__(self, input):
        super(WeightedGate, self).__init__()
        self.fc = nn.Linear(input, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate_weights = self.sigmoid(self.fc(x))
        output = gate_weights * x
        return output
    def initialize(self):
        weight_init(self)
