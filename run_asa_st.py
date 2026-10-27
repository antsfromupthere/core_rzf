import time

import torch

from common.networks import EndToEndNet, annealed_sign, hard_sign
from common.precoding import sum_rate
from common.system import (BIT_SWEEP, DEFAULT_BITS, EVAL_CHUNK, TEST_SEED, TRAIN_SEED_OFFSET,
                           base_parser, cpu_state, generate_channels, load_test_channels, load_weights, noise_power,
                           noise_std, released_bit_configs, resolve_device, save_weights, select_configs, set_seed,
                           weight_file)

NAME = 'ASA-ST'
METHOD = 'asa_st'

N_STEPS = 30000
BATCH_SIZE = 1024
LR, LR_FINAL = 1e-3, 1e-5
SLOPE_START, SLOPE_END = 0.5, 10.0
PRINT_EVERY = 2000


@torch.no_grad()
def evaluate(model, channels, snr):
    model.eval()
    std, npow = noise_std(snr), noise_power(snr)
    device = model.pilots_r.device
    total = 0.0
    for start in range(0, channels.shape[0], EVAL_CHUNK):
        h = channels[start:start + EVAL_CHUNK].to(device)
        feedback = hard_sign(model.encode(model.observe(h, std)))
        total += float(sum_rate(h, model.precode(feedback), npow).sum())
    model.train()
    return total / channels.shape[0]


def train(bits, snr, seed, device):
    std, npow = noise_std(snr), noise_power(snr)
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
        feedback = annealed_sign(model.encode(model.observe(h, std)), slope)
        rate = sum_rate(h, model.precode(feedback), npow).mean()
        optimizer.zero_grad(set_to_none=True)
        (-rate).backward()
        optimizer.step()
        scheduler.step()
        if step % PRINT_EVERY == 0:
            print('  step %5d  train %.4f  slope %.3f  %.0fs' % (step, float(rate), slope, time.time() - start),
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
        print('%s  B = %d  SNR = %g dB  sum rate %.4f bps/Hz' % (NAME, bits, snr, evaluate(model, test_channels, snr)),
              flush=True)


if __name__ == '__main__':
    main()
