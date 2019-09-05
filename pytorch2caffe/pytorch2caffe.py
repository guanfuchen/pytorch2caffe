import torch
from torch import nn
import torch.nn.functional as F
import traceback


def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    print('_conv2d----in----')
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    return x


class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace function
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        out = self.obj(self.raw, *args, **kwargs)
        return out


F.conv2d = Rp(F.conv2d, _conv2d)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 28, 28)
    net = Net()
    out = net(x)
