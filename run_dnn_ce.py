import time

import torch

from common.networks import EndToEndNet, annealed_sign, hard_sign
from common.precoding import rzf_precoder, sum_rate
from common.system import (BIT_SWEEP, DEFAULT_BITS, EVAL_CHUNK, K, P, TEST_SEED, TRAIN_SEED_OFFSET,
                           base_parser, cpu_state, generate_channels, load_test_channels, load_weights, noise_power,
                           noise_std, released_bit_configs, resolve_device, save_weights, select_configs, set_seed,
                           weight_file)

NAME = 'DNN-CE'
METHOD = 'dnn_ce'

N_STEPS = 30000
BATCH_SIZE = 1024
LR, LR_FINAL = 1e-3, 1e-5
SLOPE_START, SLOPE_END = 0.5, 10.0
PRINT_EVERY = 2000


def reconstruct(model, feedback):
    hr, hi = model.bs_heads(feedback)
    return torch.complex(hr, hi)


def nmse(h, h_hat):
    return ((h - h_hat).abs().square().sum(dim=-1) / h.abs().square().sum(dim=-1).clamp_min(1e-12)).mean()


@torch.no_grad()
def evaluate(model, channels, snr):
    model.eval()
    std, npow = noise_std(snr), noise_power(snr)
    device = model.pilots_r.device
    error, rate = 0.0, 0.0
    for start in range(0, channels.shape[0], EVAL_CHUNK):
        h = channels[start:start + EVAL_CHUNK].to(device)
        h_hat = reconstruct(model, hard_sign(model.encode(model.observe(h, std))))
        error += float(nmse(h, h_hat)) * h.shape[0]
        rate += float(sum_rate(h, rzf_precoder(h_hat, K * npow / P), npow).sum())
    model.train()
    return error / channels.shape[0], rate / channels.shape[0]


def train(bits, snr, seed, device):
    std = noise_std(snr)
    set_seed(seed)
    model = EndToEndNet(bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_FINAL / LR) ** (1.0 / (N_STEPS - 1)))
    set_seed(seed + TRAIN_SEED_OFFSET)
    model.train()
    start = time.time()
    for step in range(1, N_STEPS + 1):
        slope = SLOPE_START * (SLOPE_END / SLOPE_START) ** (step / N_STEPS)
        h = generate_channels(BATCH_SIZE, device)
        loss = nmse(h, reconstruct(model, annealed_sign(model.encode(model.observe(h, std)), slope)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % PRINT_EVERY == 0:
            print('  step %5d  train NMSE %.4f  slope %.3f  %.0fs' % (step, float(loss), slope, time.time() - start),
                  flush=True)
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
            model = load_weights(EndToEndNet(bits), args.weights or weight_file(METHOD, name)).to(device)
        set_seed(TEST_SEED)
        error, rate = evaluate(model, test_channels, snr)
        print('%s  B = %d  SNR = %g dB  sum rate %.4f bps/Hz  NMSE %.4f' % (NAME, bits, snr, rate, error), flush=True)


if __name__ == '__main__':
    main()
