import math
import time

import torch
from torch import nn

from common.networks import BS_WIDTHS, UE_WIDTHS, HiddenNetwork, glorot_init
from common.precoding import normalize_precoder, rzf_precoder, sum_rate
from common.system import (DEFAULT_BITS, EVAL_CHUNK, K, L, M, P, TEST_SEED, TRAIN_SEED_OFFSET,
                           base_parser, cpu_state, generate_channels, load_test_channels, load_weights, noise_power,
                           noise_std, released_bit_configs, resolve_device, save_weights, select_configs, set_seed,
                           weight_file)

NAME = 'CoRe-RZF'
METHOD = 'core_rzf'
ALLOCATION = {
    5: (5, 0, 0, 0),
    10: (7, 2, 1, 0),
    15: (7, 6, 2, 0),
    20: (8, 7, 3, 2),
    25: (9, 8, 4, 4),
    30: (10, 10, 5, 5),
}
SIN_THETA_MAX = 0.5
GRID_BITS = 10
GRID = 2 ** GRID_BITS
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
A_MIN, A_MAX = -3.0, 1.0
A_FIXED = -1.5
RZF_BETA = K / M

N_STEPS = 30000
BATCH_SIZE = 1024
LR_UE, LR_PILOT, LR_BS = 1e-3, 1e-4, 1e-3
LR_DECAY_STEPS = 50000
LR_END_RATIO = 1e-2
PRINT_EVERY = 2000


def array_response(s):
    n = torch.arange(M, dtype=s.dtype, device=s.device)
    phase = math.pi * s.unsqueeze(-1) * n
    return torch.complex(torch.cos(phase), torch.sin(phase))


def direction_level(index, bits):
    return -SIN_THETA_MAX + (index.to(torch.float32) + 0.5) * (2.0 * SIN_THETA_MAX / (2 ** bits))


def candidate_grid(device):
    return direction_level(torch.arange(GRID, device=device), GRID_BITS)


def score_matrix(s):
    lags = torch.arange(1, M, dtype=torch.float32, device=s.device)
    arg = math.pi * s[:, None] * lags[None, :]
    return torch.cat((torch.ones(s.shape[0], 1, device=s.device), 2 * torch.cos(arg), 2 * torch.sin(arg)), dim=1) / M


def quantize(x, lo, hi, bits):
    cells = 2 ** bits
    return torch.floor((x - lo) / (hi - lo) * cells).long().clamp_(0, cells - 1)


def dequantize(index, lo, hi, bits):
    return lo + (index.to(torch.float32) + 0.5) * (hi - lo) / (2 ** bits)


def relative_gain_table(b_ph, b_a):
    cells = torch.arange(2 ** (b_ph + b_a))
    phase = dequantize(cells >> b_a, PHASE_MIN, PHASE_MAX, b_ph)
    if b_a > 0:
        log_amp = dequantize(cells & ((1 << b_a) - 1), A_MIN, A_MAX, b_a)
    else:
        log_amp = torch.full_like(phase, A_FIXED)
    return torch.complex(torch.exp(log_amp) * torch.cos(phase), torch.exp(log_amp) * torch.sin(phase))


def pilot_response(responses, pilots):
    return torch.einsum('...m,lm->...l', responses, pilots)


def ls_fit(y, d, ridge):
    gram = torch.einsum('...tl,...ul->...tu', torch.conj(d), d)
    gram = gram + ridge * torch.eye(d.shape[-2], dtype=gram.dtype, device=gram.device)
    r = torch.einsum('...tl,...l->...t', torch.conj(d), y)
    c = torch.linalg.solve(gram, r.unsqueeze(-1)).squeeze(-1)
    return c, y - torch.einsum('...t,...tl->...l', c, d)


def field_widths(bits):
    return [(field, width) for field, width in zip(('d', 'r', 'ph', 'a'), ALLOCATION[bits]) if width > 0]


def pack_bits(indices, bits):
    chunks = []
    for field, width in field_widths(bits):
        shifts = torch.arange(width - 1, -1, -1, device=indices[field].device)
        chunks.append(((indices[field].unsqueeze(-1) >> shifts) & 1).to(torch.float32))
    return torch.cat(chunks, dim=-1)


def unpack_bits(message, bits):
    message = message.round().long()
    indices, start = {}, 0
    for field, width in field_widths(bits):
        weights = 1 << torch.arange(width - 1, -1, -1, device=message.device)
        indices[field] = (message[..., start:start + width] * weights).sum(dim=-1)
        start += width
    return indices


class CoReRZF(nn.Module):
    def __init__(self, bits, ridge):
        super().__init__()
        self.bits = int(bits)
        self.b_d, self.b_r, self.b_ph, self.b_a = ALLOCATION[self.bits]
        self.ridge = float(ridge)
        scale = math.sqrt(P / (2 * M))
        self.pilots_r = nn.Parameter(torch.randn(L, M) * scale)
        self.pilots_i = nn.Parameter(torch.randn(L, M) * scale)
        grid = candidate_grid(torch.device('cpu'))
        self.register_buffer('grid_response', array_response(grid), persistent=False)
        self.register_buffer('score_map', score_matrix(grid), persistent=False)
        self.register_buffer('gain_table', relative_gain_table(self.b_ph, self.b_a), persistent=False)
        self.ue_hidden = nn.ModuleList(HiddenNetwork(2 * L, UE_WIDTHS, input_bn=True) for _ in range(K))
        self.ue_out = nn.ModuleList(nn.Linear(self.ue_hidden[k].output_size, 2 * M - 1) for k in range(K))
        for layer in self.ue_out:
            glorot_init(layer)
        self.bs_hidden = HiddenNetwork(4 * K * M, BS_WIDTHS, input_bn=False)
        self.real_head = nn.Linear(self.bs_hidden.output_size, M * K)
        self.imag_head = nn.Linear(self.bs_hidden.output_size, M * K)
        for head in (self.real_head, self.imag_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def parameter_groups(self):
        ue = [p for m in (self.ue_hidden, self.ue_out) for p in m.parameters()]
        bs = [p for m in (self.bs_hidden, self.real_head, self.imag_head) for p in m.parameters()]
        return ue, [self.pilots_r, self.pilots_i], bs

    def pilots(self):
        power = (self.pilots_r.square() + self.pilots_i.square()).sum(dim=1, keepdim=True)
        scale = math.sqrt(P) / power.clamp_min(1e-12).sqrt()
        return torch.complex(self.pilots_r * scale, self.pilots_i * scale)

    def observe(self, h, noise_std):
        y = torch.einsum('lm,nkm->nkl', self.pilots(), h)
        if noise_std > 0.0:
            y = y + torch.complex(torch.randn_like(y.real), torch.randn_like(y.imag)) * float(noise_std)
        return y

    def scores(self, y):
        x = torch.cat((y.real, y.imag), dim=-1)
        return torch.stack([self.ue_out[k](self.ue_hidden[k](x[:, k])) @ self.score_map.t() for k in range(K)], dim=1)

    def labels(self, h):
        gain = torch.einsum('gm,nkm->nkg', torch.conj(self.grid_response), h).abs().square()
        return gain / gain.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def feedback_indices(self, y, learned):
        if self.b_r == 0:
            return {'d': learned >> (GRID_BITS - self.b_d)}
        pilots = self.pilots()
        candidates = pilot_response(array_response(candidate_grid(y.device)), pilots)
        norm = candidates.abs().square().sum(dim=-1) + self.ridge
        _, residual = ls_fit(y, candidates[learned].unsqueeze(-2), self.ridge)
        second = (torch.einsum('...l,cl->...c', residual, torch.conj(candidates)).abs().square() / norm).argmax(dim=-1)
        coef, _ = ls_fit(y, torch.stack((candidates[learned], candidates[second]), dim=-2), self.ridge)
        swap = coef[..., 1].abs() > coef[..., 0].abs()
        dominant = torch.where(swap, second, learned)
        refinement = torch.where(swap, learned, second)
        directions = direction_level(torch.stack((dominant, refinement), dim=-1), GRID_BITS)
        coef, _ = ls_fit(y, pilot_response(array_response(directions), pilots), self.ridge)
        c_d = torch.where(coef[..., 0].abs() < 1e-9, torch.ones_like(coef[..., 0]), coef[..., 0])
        rho = coef[..., 1] / c_d
        indices = {'d': dominant >> (GRID_BITS - self.b_d), 'r': refinement >> (GRID_BITS - self.b_r),
                   'ph': quantize(torch.angle(rho), PHASE_MIN, PHASE_MAX, self.b_ph)}
        if self.b_a > 0:
            log_amp = torch.log(rho.abs().clamp_min(1e-12)).clamp(A_MIN, A_MAX - 1e-6)
            indices['a'] = quantize(log_amp, A_MIN, A_MAX, self.b_a)
        return indices

    def encode(self, h, noise_std):
        y = self.observe(h, noise_std)
        scores = self.scores(y)
        indices = self.feedback_indices(y.detach(), scores.detach().argmax(dim=-1))
        return pack_bits(indices, self.bits), scores

    def channel_proxy(self, message):
        indices = unpack_bits(message, self.bits)
        h = array_response(direction_level(indices['d'], self.b_d))
        if self.b_r > 0:
            cell = indices['ph'] * (2 ** self.b_a) + indices['a'] if self.b_a > 0 else indices['ph']
            h = h + self.gain_table[cell].unsqueeze(-1) * array_response(direction_level(indices['r'], self.b_r))
        return h / h.abs().square().sum(dim=-1, keepdim=True).sqrt()

    def base_precoder(self, message):
        h_hat = self.channel_proxy(message)
        return h_hat, rzf_precoder(h_hat, RZF_BETA)

    def correct(self, h_hat, v_rzf):
        n = h_hat.shape[0]
        x = torch.cat((h_hat.real.reshape(n, -1), h_hat.imag.reshape(n, -1),
                       v_rzf.real.reshape(n, -1), v_rzf.imag.reshape(n, -1)), dim=-1)
        hidden = self.bs_hidden(x)
        dr = self.real_head(hidden).reshape(n, K, M).permute(0, 2, 1)
        di = self.imag_head(hidden).reshape(n, K, M).permute(0, 2, 1)
        return normalize_precoder(torch.complex(v_rzf.real + dr, v_rzf.imag + di))

    def forward(self, h, noise_std, noise_pow):
        message, scores = self.encode(h, noise_std)
        loss_dir = -(self.labels(h) * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        with torch.no_grad():
            h_hat, v_rzf = self.base_precoder(message)
        rate = sum_rate(h, self.correct(h_hat, v_rzf), noise_pow).mean()
        return loss_dir, rate


@torch.no_grad()
def evaluate(model, channels, snr):
    model.eval()
    std, npow = noise_std(snr), noise_power(snr)
    device = model.pilots_r.device
    total = 0.0
    for start in range(0, channels.shape[0], EVAL_CHUNK):
        h = channels[start:start + EVAL_CHUNK].to(device)
        message, _ = model.encode(h, std)
        h_hat, v_rzf = model.base_precoder(message)
        total += float(sum_rate(h, model.correct(h_hat, v_rzf), npow).sum())
    model.train()
    return total / channels.shape[0]


def train(bits, snr, seed, device):
    std, npow = noise_std(snr), noise_power(snr)
    set_seed(seed)
    model = CoReRZF(bits, npow).to(device)
    ue, pilots, bs = model.parameter_groups()
    optimizer = torch.optim.Adam([{'params': ue, 'lr': LR_UE}, {'params': pilots, 'lr': LR_PILOT},
                                  {'params': bs, 'lr': LR_BS}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_END_RATIO ** (1.0 / (LR_DECAY_STEPS - 1)))
    set_seed(seed + TRAIN_SEED_OFFSET)
    model.train()
    start = time.time()
    for step in range(1, N_STEPS + 1):
        h = generate_channels(BATCH_SIZE, device)
        loss_dir, rate = model(h, std, npow)
        optimizer.zero_grad(set_to_none=True)
        (loss_dir - rate).backward()
        optimizer.step()
        scheduler.step()
        if step % PRINT_EVERY == 0:
            print('  step %5d  train %.4f  CE %.4f  %.0fs' % (step, float(rate), float(loss_dir), time.time() - start),
                  flush=True)
    return model


def main():
    parser = base_parser()
    parser.add_argument('--bits', type=int, choices=sorted(ALLOCATION))
    args = parser.parse_args()
    device = resolve_device(args.device)
    test_channels = load_test_channels()
    for bits, snr in select_configs(args, 'bits', DEFAULT_BITS, released_bit_configs()):
        name = 'B%d_SNR%gdB' % (bits, snr)
        if args.train:
            model = train(bits, snr, args.seed, device)
            save_weights(cpu_state(model), weight_file(METHOD, name, trained=True))
        else:
            model = load_weights(CoReRZF(bits, noise_power(snr)), args.weights or weight_file(METHOD, name)).to(device)
        set_seed(TEST_SEED)
        print('%s  B = %d  SNR = %g dB  sum rate %.4f bps/Hz' % (NAME, bits, snr, evaluate(model, test_channels, snr)),
              flush=True)


if __name__ == '__main__':
    main()
