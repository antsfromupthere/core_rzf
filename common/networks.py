import math

import torch
from torch import nn

from common.system import K, L, M, P

UE_WIDTHS = (1024, 512, 256)
BS_WIDTHS = (1024, 512, 512)


def glorot_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class HiddenNetwork(nn.Module):
    def __init__(self, input_size, widths, input_bn):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size, eps=1e-3, momentum=0.01) if input_bn else nn.Identity()
        layers, previous = [], input_size
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.BatchNorm1d(width, eps=1e-3, momentum=0.01)))
            previous = width
        self.layers = nn.Sequential(*layers)
        self.output_size = previous
        self.apply(glorot_init)

    def forward(self, x):
        return self.layers(self.input_bn(x))


def annealed_sign(logits, slope):
    soft = torch.tanh(float(slope) * logits)
    return soft + (torch.sign(logits) - soft).detach()


def hard_sign(logits):
    b = torch.sign(logits)
    return torch.where(b == 0, torch.ones_like(b), b)


class EndToEndNet(nn.Module):
    def __init__(self, latent_width):
        super().__init__()
        self.latent_width = int(latent_width)
        antenna = torch.arange(M, dtype=torch.float32)
        rows = torch.arange(L, dtype=torch.float32) * math.ceil(M / L)
        phase = -2.0 * math.pi * rows[:, None] * antenna[None, :] / M
        scale = math.sqrt(P / M)
        self.pilots_r = nn.Parameter(torch.cos(phase) * scale)
        self.pilots_i = nn.Parameter(torch.sin(phase) * scale)
        self.ue_hidden = nn.ModuleList(HiddenNetwork(2 * L, UE_WIDTHS, input_bn=True) for _ in range(K))
        self.ue_out = nn.ModuleList(nn.Linear(self.ue_hidden[k].output_size, self.latent_width) for k in range(K))
        for layer in self.ue_out:
            glorot_init(layer)
        self.bs_hidden = HiddenNetwork(K * self.latent_width, BS_WIDTHS, input_bn=False)
        self.real_head = nn.Linear(self.bs_hidden.output_size, M * K)
        self.imag_head = nn.Linear(self.bs_hidden.output_size, M * K)
        glorot_init(self.real_head)
        glorot_init(self.imag_head)

    def pilots(self):
        power = (self.pilots_r.square() + self.pilots_i.square()).sum(dim=1, keepdim=True)
        scale = math.sqrt(P) / power.clamp_min(1e-12).sqrt()
        return self.pilots_r * scale, self.pilots_i * scale

    def observe(self, channels, noise_std):
        xr, xi = self.pilots()
        hr, hi = channels.real, channels.imag
        yr = torch.einsum('lm,nkm->nkl', xr, hr) - torch.einsum('lm,nkm->nkl', xi, hi)
        yi = torch.einsum('lm,nkm->nkl', xr, hi) + torch.einsum('lm,nkm->nkl', xi, hr)
        y = torch.cat((yr, yi), dim=-1)
        if noise_std > 0.0:
            y = y + float(noise_std) * torch.randn_like(y)
        return y

    def encode(self, y):
        return torch.stack([self.ue_out[k](self.ue_hidden[k](y[:, k])) for k in range(K)], dim=1)

    def bs_heads(self, latent):
        n = latent.shape[0]
        hidden = self.bs_hidden(latent.reshape(n, -1))
        return self.real_head(hidden).reshape(n, K, M), self.imag_head(hidden).reshape(n, K, M)

    def precode(self, latent):
        vr, vi = self.bs_heads(latent)
        vr, vi = vr.permute(0, 2, 1), vi.permute(0, 2, 1)
        power = (vr.square() + vi.square()).sum(dim=(-2, -1), keepdim=True)
        scale = math.sqrt(P) / power.clamp_min(1e-12).sqrt()
        return torch.complex(vr * scale, vi * scale)
