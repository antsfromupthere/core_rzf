import argparse
import math
import os
import random

import numpy as np
import torch

M = 64
K = 2
L = 8
LP = 2
P = 1.0
AOD_MAX_DEG = 30.0

BIT_SWEEP = (5, 10, 15, 20, 25, 30)
SNR_SWEEP = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
DEFAULT_BITS = 20
DEFAULT_SNR = 10.0

VAL_SEED_OFFSET = 10000
TRAIN_SEED_OFFSET = 20000
TEST_SEED = 70000
EVAL_CHUNK = 2048

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SET = os.path.join(ROOT, 'data', 'Data_test_K2M64Lp2L8_withParams.mat')
WEIGHTS_DIR = os.path.join(ROOT, 'weights')
CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints')


def resolve_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noise_power(snr_db):
    return P / (10.0 ** (snr_db / 10.0))


def noise_std(snr_db):
    return math.sqrt(noise_power(snr_db) / 2.0)


def generate_channels(n, device):
    shape = (n, LP, K)
    gain_r = torch.randn(shape, device=device) * (1.0 / math.sqrt(2))
    gain_i = torch.randn(shape, device=device) * (1.0 / math.sqrt(2))
    u = torch.rand(shape, device=device)
    aod = (2.0 * u - 1.0) * AOD_MAX_DEG * (math.pi / 180.0)
    gain = torch.complex(gain_r, gain_i)
    n_ant = torch.arange(M, dtype=torch.float32, device=device)
    phase = math.pi * torch.sin(aod).unsqueeze(-1) * n_ant
    response = torch.complex(torch.cos(phase), torch.sin(phase))
    return (gain.unsqueeze(-1) * response).sum(dim=1) * (1.0 / math.sqrt(LP))


def load_test_channels(device=None):
    import scipy.io as sio

    mat = sio.loadmat(TEST_SET)
    h = mat['hR_act_test_Final'] + 1j * mat['hI_act_test_Final']
    h = torch.from_numpy(np.ascontiguousarray(np.transpose(h, (0, 2, 1))).astype(np.complex64))
    return h.to(device) if device is not None else h


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def weight_file(method, name, trained=False):
    return os.path.join(CHECKPOINT_DIR if trained else WEIGHTS_DIR, method, name + '.pt')


def load_weights(module, path):
    module.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return module


def save_weights(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--weights')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device')
    return parser


def select_configs(args, key, default, released):
    value = getattr(args, key)
    if value is None and args.snr is None and not args.train and not args.weights:
        return released
    return [(default if value is None else value, DEFAULT_SNR if args.snr is None else args.snr)]


def released_bit_configs():
    return [(b, DEFAULT_SNR) for b in BIT_SWEEP] + [(DEFAULT_BITS, s) for s in SNR_SWEEP if s != DEFAULT_SNR]
