import math

import torch

from common.system import P


def complex_einsum(ar, ai, br, bi, eq):
    rr = torch.einsum(eq, ar, br) - torch.einsum(eq, ai, bi)
    ii = torch.einsum(eq, ar, bi) + torch.einsum(eq, ai, br)
    return rr, ii


def sum_rate(channels, precoders, noise_pow):
    er, ei = complex_einsum(channels.real, channels.imag, precoders.real, precoders.imag, 'nkm,nmj->nkj')
    power = er.square() + ei.square()
    desired = power.diagonal(dim1=-2, dim2=-1)
    interference = power.sum(dim=-1) - desired
    return torch.log2(1.0 + desired / (interference + noise_pow)).sum(dim=-1)


def normalize_precoder(v, tx_power=P):
    power = v.abs().square().sum(dim=(-2, -1), keepdim=True)
    return v * (math.sqrt(tx_power) / power.clamp_min(1e-12).sqrt())


def rzf_precoder(h, beta, tx_power=P):
    hc = torch.conj(h)
    gram = torch.einsum('nkm,njm->nkj', h, hc)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    v = torch.einsum('nkm,nkj->nmj', hc, torch.linalg.inv(gram + beta * eye))
    return normalize_precoder(v, tx_power)
