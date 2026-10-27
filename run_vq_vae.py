import math
import time

import torch
from torch import nn

from common.networks import EndToEndNet
from common.precoding import sum_rate
from common.system import (BIT_SWEEP, DEFAULT_BITS, EVAL_CHUNK, K, TEST_SEED, TRAIN_SEED_OFFSET,
                           base_parser, cpu_state, generate_channels, load_test_channels, load_weights, noise_power,
                           noise_std, released_bit_configs, resolve_device, save_weights, select_configs, set_seed,
                           weight_file)

NAME = 'VQ-VAE'
METHOD = 'vq_vae'
CODE_DIM = 256
MAX_BITS_PER_CODEBOOK = 10
COMMITMENT_BETA = 0.1
MI_WEIGHT = 1.0
MI_SIGMA = 0.05
MI_KAPPA = 16

N_STEPS = 30000
BATCH_SIZE = 1024
LR, CODEBOOK_LR, LR_FINAL = 1e-3, 1e-2, 1e-5
PRINT_EVERY = 2000


def split_bits(bits):
    count = max(1, math.ceil(bits / MAX_BITS_PER_CODEBOOK))
    base, extra = divmod(bits, count)
    return [base + 1] * extra + [base] * (count - extra)


class VQVAE(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.codebook_bits = split_bits(bits)
        self.codebooks = nn.ParameterList(
            nn.Parameter(torch.randn(2 ** b, CODE_DIM) / math.sqrt(CODE_DIM)) for _ in range(K) for b in self.codebook_bits)
        self.net = EndToEndNet(len(self.codebook_bits) * CODE_DIM)

    @torch.no_grad()
    def uniform_init(self):
        for book in self.codebooks:
            bound = 1.0 / book.shape[0]
            book.uniform_(-bound, bound)

    def quantize(self, latents, straight_through=True):
        count = len(self.codebook_bits)
        blocks = latents.reshape(latents.shape[0], K, count, CODE_DIM)
        chosen = []
        for k in range(K):
            for i in range(count):
                book = self.codebooks[k * count + i]
                nearest = (blocks[:, k, i] @ book.t() - 0.5 * book.square().sum(dim=1)).argmax(dim=1)
                chosen.append(book[nearest])
        quantized = torch.stack(chosen, dim=1).reshape(latents.shape[0], K, count * CODE_DIM)
        commitment = ((quantized.detach() - latents).square() * COMMITMENT_BETA
                      + (quantized - latents.detach()).square()).sum(dim=-1).sum(dim=-1).mean()
        if straight_through:
            quantized = latents + (quantized - latents).detach()
        return quantized, commitment


def mi_lower_bound(latents):
    n = latents.shape[0]
    pairs = MI_KAPPA * n
    first = torch.randint(0, n, (pairs,), device=latents.device)
    second = (first + torch.randint(1, n, (pairs,), device=latents.device)) % n
    squared = (latents[first] - latents[second]).square().sum(dim=-1)
    kernel = torch.exp(-squared / (2.0 * MI_SIGMA ** 2))
    return -torch.log(kernel.mean(dim=0).clamp_min(1e-12)).sum()


@torch.no_grad()
def evaluate(model, channels, snr):
    model.eval()
    std, npow = noise_std(snr), noise_power(snr)
    device = model.net.pilots_r.device
    total = 0.0
    for start in range(0, channels.shape[0], EVAL_CHUNK):
        h = channels[start:start + EVAL_CHUNK].to(device)
        quantized, _ = model.quantize(model.net.encode(model.net.observe(h, std)), straight_through=False)
        total += float(sum_rate(h, model.net.precode(quantized), npow).sum())
    model.train()
    return total / channels.shape[0]


def train(bits, snr, seed, device):
    std, npow = noise_std(snr), noise_power(snr)
    set_seed(seed)
    model = VQVAE(bits).to(device)
    optimizer = torch.optim.Adam([{'params': list(model.codebooks.parameters()), 'lr': CODEBOOK_LR},
                                  {'params': list(model.net.parameters()), 'lr': LR}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_FINAL / LR) ** (1.0 / (N_STEPS - 1)))
    set_seed(seed + TRAIN_SEED_OFFSET)
    model.uniform_init()
    model.train()
    start = time.time()
    for step in range(1, N_STEPS + 1):
        h = generate_channels(BATCH_SIZE, device)
        latents = model.net.encode(model.net.observe(h, std))
        quantized, commitment = model.quantize(latents)
        rate = sum_rate(h, model.net.precode(quantized), npow).mean()
        mi = mi_lower_bound(quantized.reshape(BATCH_SIZE, -1, CODE_DIM))
        loss = -rate - MI_WEIGHT * mi + commitment
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % PRINT_EVERY == 0:
            print('  step %5d  train %.4f  commitment %.3f  %.0fs' % (
                step, float(rate), float(commitment), time.time() - start), flush=True)
    return model


def main():
    parser = base_parser()
    parser.add_argument('--bits', type=int, choices=BIT_SWEEP)
    args = parser.parse_args()
    device = resolve_device(args.device)
    test_channels = load_test_channels()
    for bits, snr in select_configs(args, 'bits', DEFAULT_BITS, released_bit_configs()):
        name = 'B%d_SNR%gdB' % (bits, snr)
        if args.train:
            model = train(bits, snr, args.seed, device)
            save_weights(cpu_state(model), weight_file(METHOD, name, trained=True))
        else:
            model = load_weights(VQVAE(bits), args.weights or weight_file(METHOD, name)).to(device)
        set_seed(TEST_SEED)
        print('%s  B = %d  SNR = %g dB  sum rate %.4f bps/Hz' % (NAME, bits, snr, evaluate(model, test_channels, snr)),
              flush=True)


if __name__ == '__main__':
    main()
