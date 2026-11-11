"""Small-image adaptation of RepVGG, not an official RepVGG-A/B model."""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class RepBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, branches="full", deploy=False):
        super().__init__()
        self.cin, self.cout, self.stride = cin, cout, stride
        self.branches, self.deploy = branches, deploy
        if deploy:
            self.reparam = nn.Conv2d(cin, cout, 3, stride, 1, bias=True)
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                                       nn.BatchNorm2d(cout))
            self.conv1 = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False),
                                        nn.BatchNorm2d(cout))
                          if branches in ("conv1", "full") else None)
            self.identity = (nn.BatchNorm2d(cin)
                             if branches == "full" and cin == cout and stride == 1 else None)

    def forward(self, x):
        if self.deploy:
            return F.relu(self.reparam(x))
        y = self.conv3(x)
        if self.conv1 is not None:
            y = y + self.conv1(x)
        if self.identity is not None:
            y = y + self.identity(x)
        return F.relu(y)

    def _fuse(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv, bn = branch
            kernel = conv.weight
        else:
            bn = branch
            kernel = torch.zeros((self.cout, self.cin, 3, 3),
                                 dtype=bn.weight.dtype, device=bn.weight.device)
            idx = torch.arange(self.cin, device=kernel.device)
            kernel[idx, idx, 1, 1] = 1
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return kernel * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        if self.training:
            raise RuntimeError("Call eval() before fusing BatchNorm running statistics.")
        k3, b3 = self._fuse(self.conv3)
        k1, b1 = self._fuse(self.conv1)
        ki, bi = self._fuse(self.identity)
        if self.conv1 is not None:
            k1 = F.pad(k1, [1, 1, 1, 1])
        self.reparam = nn.Conv2d(self.cin, self.cout, 3, self.stride, 1, bias=True)
        self.reparam.to(device=k3.device, dtype=k3.dtype)
        self.reparam.weight.copy_(k3 + k1 + ki)
        self.reparam.bias.copy_(b3 + b1 + bi)
        del self.conv3, self.conv1, self.identity
        self.deploy = True


class SmallRepVGG(nn.Module):
    def __init__(self, num_classes=10, branches="full", deploy=False,
                 widths=(16, 32, 64, 128), depths=(1, 2, 2, 2)):
        super().__init__()
        if branches not in ("baseline", "conv1", "full"):
            raise ValueError(branches)
        blocks, cin = [], 3
        for stage, (cout, depth) in enumerate(zip(widths, depths)):
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(RepBlock(cin, cout, stride, branches, deploy))
                cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def deploy_copy(model):
    model = copy.deepcopy(model).eval()
    for module in list(model.modules()):
        if isinstance(module, RepBlock):
            module.switch_to_deploy()
    return model
