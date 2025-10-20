"""
Feature extraction backbone for GMFlow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for feature extraction"""
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm1
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    """
    CNN backbone for feature extraction
    Output feature channels: 256
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, num_output_scales=1):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(64) if norm_fn == 'batch' else nn.GroupNorm(8, 64)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, stride=1, norm_fn=norm_fn)
        self.layer2 = self._make_layer(64, 96, stride=2, norm_fn=norm_fn)
        self.layer3 = self._make_layer(96, 128, stride=2, norm_fn=norm_fn)

        # Output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, stride, norm_fn):
        layer1 = ResidualBlock(in_planes, planes, norm_fn, stride=stride)
        layer2 = ResidualBlock(planes, planes, norm_fn, stride=1)
        layers = [layer1, layer2]
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)  # [B, 64, H/2, W/2]

        x = self.layer1(x)  # [B, 64, H/2, W/2]
        x = self.layer2(x)  # [B, 96, H/4, W/4]
        x = self.layer3(x)  # [B, 128, H/8, W/8]

        x = self.conv2(x)  # [B, output_dim, H/8, W/8]

        if self.dropout is not None:
            x = self.dropout(x)

        return x
