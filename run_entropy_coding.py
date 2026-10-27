import time

import torch
from torch import nn

from common.networks import EndToEndNet
from common.precoding import sum_rate
from common.system import (DEFAULT_SNR, EVAL_CHUNK, K, SNR_SWEEP, TEST_SEED, TRAIN_SEED_OFFSET,
                           base_parser, cpu_state, generate_channels, load_test_channels, load_weights, noise_power,
                           noise_std, resolve_device, save_weights, select_configs, set_seed, weight_file)

NAME = 'Entropy Coding'
METHOD = 'entropy_coding'
NB = 16
LAMBDA_SWEEP = (5.0, 10.0, 20.0, 50.0, 62.0, 100.0, 140.0)
DEFAULT_LAMBDA = 62.0

N_STEPS = 30000
BATCH_SIZE = 1024
LR, LR_FINAL = 1e-3, 1e-5
PRINT_EVERY = 2000


class EntropyCoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EndToEndNet(NB)
        self.log_sigma = nn.Parameter(torch.zeros(K, NB))

    def code_length(self, t):
        sigma = self.log_sigma.exp().clamp_min(1e-4)
        upper = torch.special.ndtr((t + 0.5) / sigma)
        lower = torch.special.ndtr((t - 0.5) / sigma)
        return -torch.log2((upper - lower).clamp_min(1e-9))


@torch.no_grad()
def evaluate(model, channels, snr):
    model.eval()
    std, npow = noise_std(snr), noise_power(snr)
    device = model.net.pilots_r.device
    rate, overhead = 0.0, 0.0
    for start in range(0, channels.shape[0], EVAL_CHUNK):
        h = channels[start:start + EVAL_CHUNK].to(device)
        t = torch.round(model.net.encode(model.net.observe(h, std)))
        rate += float(sum_rate(h, model.net.precode(t), npow).sum())
        overhead += float(model.code_length(t).sum(dim=-1).mean(dim=-1).sum())
    model.train()
    return rate / channels.shape[0], overhead / channels.shape[0]


def train(lam, snr, seed, device):
    std, npow = noise_std(snr), noise_power(snr)
    set_seed(seed)
    model = EntropyCoding().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_FINAL / LR) ** (1.0 / (N_STEPS - 1)))
    set_seed(seed + TRAIN_SEED_OFFSET)
    model.train()
    start = time.time()
    for step in range(1, N_STEPS + 1):
        h = generate_channels(BATCH_SIZE, device)
        t = model.net.encode(model.net.observe(h, std))
        noisy = t + (torch.rand_like(t) - 0.5)
        rate = sum_rate(h, model.net.precode(noisy), npow).mean()
        overhead = model.code_length(noisy).sum(dim=-1).sum(dim=-1).mean()
        loss = overhead - lam * rate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % PRINT_EVERY == 0:
            print('  step %5d  train %.4f  overhead %.2f bits/user  %.0fs' % (
                step, float(rate), float(overhead) / K, time.time() - start), flush=True)
    return model


def main():
    parser = base_parser()
    parser.add_argument('--lam', type=float)
    args = parser.parse_args()
    device = resolve_device(args.device)
    test_channels = load_test_channels()
    released = [(lam, DEFAULT_SNR) for lam in LAMBDA_SWEEP]
    released += [(DEFAULT_LAMBDA, s) for s in SNR_SWEEP if s != DEFAULT_SNR]
    for lam, snr in select_configs(args, 'lam', DEFAULT_LAMBDA, released):
        name = 'lambda%g_SNR%gdB' % (lam, snr)
        if args.train:
            model = train(lam, snr, args.seed, device)
            save_weights(cpu_state(model), weight_file(METHOD, name, trained=True))
        else:
            model = load_weights(EntropyCoding(), args.weights or weight_file(METHOD, name)).to(device)
        set_seed(TEST_SEED)
        rate, overhead = evaluate(model, test_channels, snr)
        print('%s  lambda = %g  SNR = %g dB  sum rate %.4f bps/Hz  overhead %.2f bits/user' % (
            NAME, lam, snr, rate, overhead), flush=True)


if __name__ == '__main__':
    main()
