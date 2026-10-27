import argparse

from common.precoding import rzf_precoder, sum_rate
from common.system import K, M, P, SNR_SWEEP, load_test_channels, noise_power, resolve_device

NAME = 'PCSIT-RZF'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float)
    parser.add_argument('--device')
    args = parser.parse_args()
    h = load_test_channels(resolve_device(args.device))
    unit = h / h.abs().square().sum(dim=-1, keepdim=True).sqrt()
    for snr in SNR_SWEEP if args.snr is None else (args.snr,):
        npow = noise_power(snr)
        rate = float(sum_rate(h, rzf_precoder(unit, K * npow / (M * P)), npow).mean())
        print('%s  SNR = %g dB  sum rate %.4f bps/Hz' % (NAME, snr, rate), flush=True)


if __name__ == '__main__':
    main()
