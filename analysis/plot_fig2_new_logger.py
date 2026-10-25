#!/usr/bin/env python3
"""
Power-trace figure (fig1_power_time_series) from the 1000 Hz logger data of Campaign v2.
Run from the repository root: python3 -I plot_fig2_new_logger.py [output dir ...]
Phase windows come from the GPIO markers; the baseline is the run's value in run_summary.csv
(median of 5-25 s of each idle period, averaged).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
})

C_LINE = '#2C3E50'
C_BASE = '#6b6b6b'
C_COMM = '#E69F00'
C_TRAIN = '#009E73'
RUN = 'main_E20_rep5'


def segments(flag, t_ms):
    """Return (start_ms, end_ms) for each run of flag == 1."""
    d = np.diff(np.r_[0, flag, 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(t_ms[a], t_ms[b - 1] + 1) for a, b in zip(s, e)]


def generate_fig1(destinations):
    df = pd.read_csv(Path('data/raw_v2/step1_main') / RUN / 'power_trace.csv')
    summ = pd.read_csv('data/raw_v2/run_summary.csv').set_index('run_id').loc[RUN]
    t_ms = df['timestamp_ms'].values
    t_sec = t_ms / 1000.0
    p_mw = df['power_mW'].values
    comm = df['gpio_comm'].values
    train = df['gpio_train'].values
    p_base = float(summ['baseline_mean_mW'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 3.8),
                                   gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.62})

    # (a) full run
    ax1.plot(t_sec, p_mw, color=C_LINE, lw=0.5, alpha=0.85, rasterized=True)
    ax1.axhline(p_base, color=C_BASE, linestyle=':', lw=1.1, zorder=2)
    ax1.fill_between(t_sec, 520, 900, where=(comm == 1), color=C_COMM, alpha=0.6, zorder=3)
    ax1.fill_between(t_sec, 520, 900, where=(train == 1), color=C_TRAIN, alpha=0.6, zorder=3)
    turn = segments(((comm == 1) | (train == 1)).astype(int), t_ms)
    t_turn = (turn[0][0] + turn[-1][1]) / 2000.0
    ax1.text(15.0, 800, '30 s idle (baseline window 5-25 s)', ha='center', fontsize=7, color=C_BASE)
    ax1.text(46.0, 800, '30 s idle (baseline window 5-25 s)', ha='center', fontsize=7, color=C_BASE)
    ax1.annotate(f'Client turn ($E={int(summ.E)}$)', xy=(t_turn, 880), xytext=(t_turn + 4, 870),
                 ha='left', va='center', fontsize=7, color=C_LINE,
                 arrowprops=dict(arrowstyle='->', color=C_LINE, lw=0.8))
    ax1.set_xlim(0, t_sec[-1])
    ax1.set_ylim(530, 900)
    ax1.set_xlabel('Time (s)', labelpad=2)
    ax1.set_ylabel('Power (mW)')
    ax1.set_title(f'(a) Full run, {t_sec[-1]:.0f} s at 1 kHz', pad=4, loc='left')

    # (b) zoom on the client turn
    t0 = turn[0][0] - 50
    t1 = turn[-1][1] + 50
    m = (t_ms >= t0) & (t_ms <= t1)
    tz = t_ms[m] - t0
    ax2.plot(tz, p_mw[m], color=C_LINE, lw=0.9, marker='o', markersize=1.6,
             markerfacecolor='white', markeredgewidth=0.4, zorder=4)
    ax2.axhline(p_base, color=C_BASE, linestyle=':', lw=1.1, zorder=2)
    ax2.fill_between(tz, 520, 1150, where=(comm[m] == 1), color=C_COMM, alpha=0.35, zorder=2)
    ax2.fill_between(tz, 520, 1150, where=(train[m] == 1), color=C_TRAIN, alpha=0.35, zorder=2)

    comm_segs = segments(comm[m], tz)
    train_segs = segments(train[m], tz)
    names = ['Config', 'Download', 'Upload']
    labelled = [(n, s) for n, s in zip(names, comm_segs)] + [(f'SGD, $E={int(summ.E)}$', train_segs[0])]
    for name, (a, b) in labelled:
        ax2.text((a + b) / 2, 1.02, f'{name}, {b - a:.0f} ms', ha='center', va='bottom', fontsize=6.5, color='#333333',
                 transform=ax2.get_xaxis_transform())
    phases = sorted([s for _, s in labelled])
    for (_, b), (a, _) in zip(phases[:-1], phases[1:]):
        ax2.annotate('', xy=(b, 600), xytext=(a, 600), arrowprops=dict(arrowstyle='<->', color=C_BASE, lw=0.6))
        ax2.text((a + b) / 2, 607, f'{a - b:.0f} ms', ha='center', va='bottom', fontsize=6, color=C_BASE)

    ax2.set_xlim(0, tz[-1])
    ax2.set_ylim(535, 1130)
    ax2.set_xlabel('Time within the turn window (ms)', labelpad=2)
    ax2.set_ylabel('Power (mW)')
    ax2.set_title('(b) One client turn at 1 ms resolution', pad=14, loc='left')

    legend_elements = [
        plt.Line2D([0], [0], color=C_LINE, lw=0.9, marker='o', markersize=2.5, markerfacecolor='white', label='Power (INA238, 1 kHz)'),
        plt.Line2D([0], [0], color=C_BASE, linestyle=':', lw=1.1, label=f'Baseline $P_{{\\rm base}}$ = {p_base:.1f} mW'),
        Patch(facecolor=C_COMM, alpha=0.5, edgecolor='none', label='Communication marker'),
        Patch(facecolor=C_TRAIN, alpha=0.5, edgecolor='none', label='Training marker'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

    for d in destinations:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'fig1_power_time_series.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[OK] fig1_power_time_series')


if __name__ == '__main__':
    generate_fig1(sys.argv[1:] or ['overleaf_project/overleaf/figures'])
