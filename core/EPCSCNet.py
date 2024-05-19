import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import numpy as np


class EPCSCNet(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=11, stride=8, num_iter=12, elastic_type='impgelu', mean_estimate=False) -> None:
        super().__init__()

        self.mean_estimate = mean_estimate
        self.num_iter = num_iter
        self.stride = stride

        self.init = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, bias=False),
            torch.nn.ReLU(inplace=True),
            LayerNorm(out_ch),
            ResBlock(out_ch, out_ch),
        )
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, bias=False)
        self.convT = torch.nn.ConvTranspose2d(out_ch, in_ch, kernel_size, stride=stride, bias=False)
        self.decoder = torch.nn.ConvTranspose2d(out_ch, in_ch, kernel_size, stride=stride, bias=False)

        self.elastic = torch.nn.ModuleList()
        self.active = torch.nn.ModuleList()

        elastic_type = elastic_type.lower()
        for i in range(num_iter):
            if elastic_type == 'impgelu':
                QTQ = MLP(out_ch, out_ch, out_ch, ImpGeLU)
            elif elastic_type == "relu":
                QTQ = MLP(out_ch, out_ch, out_ch, nn.ReLU)
            elif elastic_type == "linear":
                QTQ = MLP(out_ch, out_ch, out_ch, nn.Identity)
                # QTQ = nn.Conv2d(num_kernels, num_kernels, 1, 1, 0, bias=False)
            elif elastic_type == "none":
                QTQ = ZeroOut()
            else:
                raise TypeError
            self.elastic.append(QTQ)
            self.active.append(SoftThreshold(out_ch, 1e-3))

        W = torch.clone(self.conv.weight.data)
        eigen = conv_power_method(W, img_size=[100, 100], stride=1, num_iters=100)
        W /= eigen**0.5

        self.init[0].weight.data = torch.clone(W)
        self.decoder.weight.data = torch.clone(W)
        self.convT.weight.data = torch.clone(W)
        self.conv.weight.data = torch.clone(W)

    def forward(self, I: torch.Tensor):
        shape = I.shape
        mean = 0.5 if not self.mean_estimate else I.mean(dim=[-1, -2], keepdim=True)
        I = I - mean
        I, valids = pad_conv_stride(I, self.conv.kernel_size[0], self.stride, shift=True)
        Dx = self.conv(I)
        z = self.init(I)
        for i in range(self.num_iter):
            DDTz = self.conv(self.convT(z))
            Qz = self.elastic[i](z)
            z = z + Dx - DDTz + Qz
            z = self.active[i](z)

        rec = self.decode(z, valids, shape) + mean
        return rec

    def decode(self, z, valids, shape):
        out = self.decoder(z)
        out = torch.masked_select(out, valids.bool())
        out = out.reshape(shape[0], -1, *shape[1:])
        out = out.mean(dim=1, keepdim=False)
        return out


class ZeroOut(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return float(0.0)


@torch.jit.script
def implicit_gelu(x: torch.Tensor) -> torch.Tensor:
    PHI_x = 0.5 * (1 + torch.erf(x * 0.707106781))
    phi_x = 0.39894228040 * torch.exp(-x * x / 2)
    out = (PHI_x + x * phi_x) * x * PHI_x
    return out


class ImpGeLU(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return implicit_gelu(x)


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, activate=ImpGeLU):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = activate()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)

        W = torch.randn(hidden_features, in_features, 1, 1)
        length = W.pow(2).sum(dim=1).pow(0.5).view(-1, 1, 1, 1)
        W = W / length * 0.01

        self.fc1.weight.data = torch.clone(W)
        self.fc2.weight.data = torch.clone(W.transpose(0, 1).contiguous())

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def pad_conv_stride(I, kernel_size, stride, shift=False):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride

    if not shift:
        I_padded = F.pad(I, (left_pad, right_pad, top_pad, bot_pad), mode='reflect')
        valids = F.pad(torch.ones_like(I), (left_pad, right_pad, top_pad, bot_pad), mode='constant')
    else:
        I_padded = []
        valids = []
        stride_list = list(range(stride))
        mask = torch.ones_like(I)
        for row_shift, col_shift in itertools.product(stride_list, stride_list):
            I_shift = F.pad(
                I,
                pad=(left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift),
                mode='reflect',
            )
            valid = F.pad(
                mask,
                pad=(left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift),
                mode='constant',
            )
            I_padded.append(I_shift.unsqueeze(1))
            valids.append(valid.unsqueeze(1))

        I_padded = torch.cat(I_padded, dim=1)
        valids = torch.cat(valids, dim=1)
        I_padded = I_padded.reshape(-1, *I_padded.shape[2:])
        valids = valids.reshape(-1, *valids.shape[2:])

    return I_padded, valids


def conv_power_method(D, img_size, num_iters=100, stride=1, use_gpu=True):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    z = torch.zeros((D.shape[1], *img_size)).reshape(1, D.shape[1], *img_size)
    if use_gpu:
        z = z.cuda()
        D = D.cuda()

    z = F.conv2d(z, D, stride=stride)
    z = torch.randn_like(z)
    z = z / torch.norm(z.reshape(-1))
    L = None
    for _ in range(num_iters):
        Dz = F.conv_transpose2d(z, D, stride=stride)
        DTDz = F.conv2d(Dz, D, stride=stride)
        L = torch.norm(DTDz.reshape(-1))
        z = DTDz / L
    return L.item()


class SoftThreshold(nn.Module):

    def __init__(self, num_channels, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        threshold = self.threshold
        out = torch.sign(x) * torch.clamp_min(x.abs() - threshold, 0)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            LayerNorm(out_ch),
            torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            LayerNorm(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, z):
        z = self.shortcut(z) + self.main(z)
        return z


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, 1, 1))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x
