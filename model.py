from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def ring_basis(kernel_size):
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError('Geometric kernel size must be an odd integer >=3.')
    radius = kernel_size // 2
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1),
                          torch.arange(-radius, radius + 1), indexing='ij')
    d2 = x.square() + y.square()
    disk = d2 <= radius * radius
    rings = torch.stack([(d2 == r).float() for r in torch.unique(d2[disk])])
    rings = rings / torch.sqrt(rings.sum(dim=(-1, -2), keepdim=True))
    return rings, disk.float()


class RadialDepthwiseConv2d(nn.Module):
    """Disk-only depthwise filters with EXACT equal-distance ring weight sharing."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        if out_ch % in_ch:
            raise ValueError('out_ch must be divisible by in_ch for depthwise filtering.')
        basis, support = ring_basis(kernel_size)
        self.register_buffer('basis', basis)
        self.register_buffer('support', support)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.theta = nn.Parameter(torch.empty(out_ch, len(basis)))
        nn.init.normal_(self.theta, std=math.sqrt(2.0 / len(basis)))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def effective_weight(self):
        return torch.einsum('cr,rxy->cxy', self.theta, self.basis).unsqueeze(1)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias,
                        padding=self.kernel_size // 2, groups=self.in_ch)


class RadialConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([RadialDepthwiseConv2d(in_ch, out_ch, k) for k in (3, 5, 7)])
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.post_bn = nn.BatchNorm2d(out_ch)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fuse(torch.cat([conv(x) for conv in self.branches], dim=1))
        return self.post_relu(self.post_bn(y + x)) if x.shape[1] == y.shape[1] else y


class OriginalMultiScaleDepthwiseBlock(nn.Module):
    """Original unconstrained block: used only when mode='combined' after the requested naming swap."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(in_ch, out_ch, k, padding=k // 2, groups=in_ch) for k in (3, 5, 7)])
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.post_bn = nn.BatchNorm2d(out_ch)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fuse(torch.cat([conv(x) for conv in self.branches], dim=1))
        return self.post_relu(self.post_bn(y + x)) if x.shape[1] == y.shape[1] else y


class CircularRingSpatialFilter(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        basis, support = ring_basis(kernel_size)
        self.register_buffer('basis', basis)
        self.register_buffer('support', support)
        self.kernel_size = kernel_size
        self.theta = nn.Parameter(torch.empty(2, len(basis)))
        nn.init.normal_(self.theta, std=0.15)
        self.bias = nn.Parameter(torch.zeros(1))

    def effective_weight(self):
        return torch.einsum('cr,rxy->cxy', self.theta, self.basis).unsqueeze(0)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias, padding=self.kernel_size // 2)


class CircularAttentionGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hid = max(4, channels // reduction)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hid, 1), nn.ReLU(inplace=True),
                                          nn.Conv2d(hid, channels, 1), nn.Sigmoid())
        self.circular_spatial = CircularRingSpatialFilter(7)

    def forward(self, x):
        z = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        spatial = torch.sigmoid(self.circular_spatial(z))
        return x * self.channel_gate(x) * spatial


class OriginalChannelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        h = max(channels // 8, 1)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, h, 1), nn.ReLU(inplace=True),
                                          nn.Conv2d(h, channels, 1), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=5, padding=2, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) * self.spatial_gate(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class GeometryRCUNet(nn.Module):
    """U-Net topology with requested experiment naming: radial, combined, circular."""
    def __init__(self, mode='radial', base=64, dropout=0.2):
        super().__init__()
        if mode not in ('radial', 'combined', 'circular'):
            raise ValueError('MODE must be radial, combined, or circular.')
        # Requested label swap: new circular == old combined; new combined == old circular.
        conv = RadialConvBlock if mode in ('radial', 'circular') else OriginalMultiScaleDepthwiseBlock
        attention = CircularAttentionGate if mode in ('combined', 'circular') else OriginalChannelSpatialAttention
        self.mode = mode
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = conv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_conv = conv(base * 4, base * 8)
        self.bottleneck_attn = attention(base * 8)
        self.dropout = nn.Dropout2d(dropout)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('Height and width must be divisible by 8.')
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.dropout(self.bottleneck_attn(self.bottleneck_conv(self.pool(c))))
        y = self.dec3(torch.cat([self.up3(d), c], dim=1))
        y = self.dec2(torch.cat([self.up2(y), b], dim=1))
        y = self.dec1(torch.cat([self.up1(y), a], dim=1))
        return self.head(y)
