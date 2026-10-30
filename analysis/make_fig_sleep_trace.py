#!/usr/bin/env python3
"""Campaign 4 figure: measured board power over one sleep cycle and one deep-sleep wake-up.
Usage: python3 -I make_fig_sleep_trace.py <session dir with power_trace.csv> <data_summary dir> <figure dir> [cycle]
Writes <figure dir>/fig_sleep_trace.pdf."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sess, ds, fd = sys.argv[1], sys.argv[2], sys.argv[3]
CYC = int(sys.argv[4]) if len(sys.argv) > 4 else 7
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'font.family': 'serif', 'pdf.fonttype': 42})
BL, OR, GR, RD, AM, PU = '#0072B2', '#D55E00', '#6b6b6b', '#009E73', '#E69F00', '#CC79A7'

import os
trace = sess + '/power_trace.csv'
df = pd.read_csv(trace if os.path.exists(trace) else trace + '.gz', usecols=['timestamp_ms', 'power_mW', 'raw_phase_code'])
t, p, code = df.timestamp_ms.to_numpy(float), df.power_mW.to_numpy(float), df.raw_phase_code.to_numpy(int)
edges = np.flatnonzero(np.diff(code)) + 1
R = [(int(code[s]), int(s), int(e) - 1) for s, e in zip(np.r_[0, edges], np.r_[edges, len(code)])]
pul = lambda c: [r for r in R if r[0] == c and 10 <= r[2] - r[1] + 1 <= 40]
hold = lambda c: [r for r in R if r[0] == c and r[2] - r[1] + 1 >= 100]
k = CYC - 1
a = pul(1)[2 * k][1] - 1500
b = hold(5)[k][2] + 1500
S = json.load(open(ds + '/sleep_campaign.json'))['summary']

fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.9), gridspec_kw=dict(height_ratios=[1.35, 1]))
# (a) one cycle, 10 ms means, log power
w = 10
n = (b - a) // w
tt = (t[a:a + n * w].reshape(n, w).mean(1) - t[a]) / 1000
pp = p[a:a + n * w].reshape(n, w).mean(1)
ax[0].semilogy(tt, pp, color='#333333', lw=0.5)
spans = [('Standby', pul(1)[2 * k], pul(1)[2 * k + 1], GR, S['P_standby_mW']['mean']),
         ('Modem\nsleep', pul(2)[2 * k], pul(2)[2 * k + 1], PU, S['P_modem_mW']['mean']),
         ('Light\nsleep', pul(3)[2 * k], pul(3)[2 * k + 1], AM, S['P_light_mW']['mean'])]
d4, h5 = pul(4)[k], hold(5)[k]
for lab, s0, s1, col, pm in spans + [('Deep\nsleep', d4, (None, h5[1] - 55, None), RD, S['P_deep_mW']['mean'])]:
    x0, x1 = (t[s0[2]] - t[a]) / 1000, (t[s1[1]] - t[a]) / 1000
    ax[0].axvspan(x0, x1, color=col, alpha=0.13, lw=0)
    ax[0].text((x0 + x1) / 2, 1500, lab, ha='center', va='top', fontsize=6.5, color='#222222', linespacing=1.0)
    ax[0].text((x0 + x1) / 2, pm * 0.62, f'{pm:.1f}' if pm < 100 else f'{pm:.0f}', ha='center', va='top', fontsize=6.3, color=col)
for r, lab in [(hold(6)[k], 'wake'), (h5, 'wake')]:
    ax[0].annotate(lab, xy=((t[r[1]] - t[a]) / 1000, 900), xytext=((t[r[1]] - t[a]) / 1000, 2600), ha='center', fontsize=6,
                   arrowprops=dict(arrowstyle='-', lw=0.5, color=GR), color=GR, annotation_clip=False)
ax[0].set_ylim(5, 3000); ax[0].set_xlim(0, tt[-1])
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('Board power (mW)')
ax[0].set_title(f'(a) One measured cycle (cycle {CYC}); plateau means in mW', loc='left', fontsize=7)

# (b) deep-sleep wake-up at 1 ms
fl = np.median(p[d4[2] + 1000:h5[1] - 3000])
j = d4[2] + 1000
while not (p[j:j + 5] > fl + 50).all():
    j += 1
z0, z1 = j - 150, h5[2] + 250
x = t[z0:z1] - t[j]
ax[1].plot(x, p[z0:z1], color='#333333', lw=0.45)
ax[1].axvspan(0, t[h5[1]] - t[j], color=BL, alpha=0.15, lw=0)
ax[1].axvspan(t[h5[1]] - t[j], t[h5[2]] - t[j], color=OR, alpha=0.12, lw=0)
ax[1].text((t[h5[1]] - t[j]) / 2, 1380, 'boot', ha='center', fontsize=6.3, color=BL, rotation=90, va='top')
ax[1].text((t[h5[1]] + t[h5[2]]) / 2 - t[j], 1380, 'Wi-Fi start, scan, association, DHCP', ha='center', va='top', fontsize=6.3, color=OR)
E = (getattr(np, 'trapezoid', None) or np.trapz)(p[j:h5[2] + 1], t[j:h5[2] + 1] / 1000)
ax[1].text(0.99, 0.19, f'this wake-up: {E:.0f} mJ in {t[h5[2]] - t[j]:.0f} ms\n'
           f'20 cycles: {S["E_wake_deep_mJ"]["mean"]:.0f} mJ '
           f'[{S["E_wake_deep_mJ"]["ci95"][0]:.0f}, {S["E_wake_deep_mJ"]["ci95"][1]:.0f}]',
           transform=ax[1].transAxes, ha='right', va='center', fontsize=6.3,
           bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
ax[1].set_ylim(0, 1450); ax[1].set_xlim(x[0], x[-1])
ax[1].set_xlabel('Time after the deep-sleep timer expires (ms)'); ax[1].set_ylabel('Board power (mW)')
ax[1].set_title('(b) Deep-sleep wake-up, 1 ms samples', loc='left', fontsize=7)
plt.tight_layout(h_pad=0.6)
plt.savefig(fd + '/fig_sleep_trace.pdf'); plt.savefig(fd + '/fig_sleep_trace.png', dpi=250); plt.close()
print('ok', E)
