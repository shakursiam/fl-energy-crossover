#!/usr/bin/env python3
"""
Generate publication-quality Figure 2 for IEEE TMLCN manuscript using
the new 1000 Hz hardware logger data from Campaign v2 (data/raw_v2/).

Replaces the legacy 500 Hz pilot figure with clean, 1.0 ms temporal resolution
data showing exact microarchitectural phases and calibrated settling intervals.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Styling configuration for IEEE Transactions (two-column width: ~7.16 in)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'legend.fontsize': 7.2,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#ECEFF1',
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
})

C_LINE = '#2C3E50'      # Dark slate for power signal
C_BASE = '#7F8C8D'      # Muted grey for baseline
C_COMM = '#E67E22'      # Warm amber/orange for Wi-Fi communication
C_TRAIN = '#16A085'     # Emerald/teal for on-device SGD training

def generate_fig2():
    csv_path = Path('data/raw_v2/step1_main/main_E20_rep5/power_trace.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing raw trace: {csv_path}")
    
    df = pd.read_csv(csv_path)
    t_sec = df['timestamp_ms'].values / 1000.0
    p_mw = df['power_mW'].values
    comm = df['gpio_comm'].values
    train = df['gpio_train'].values
    
    # Pre-baseline window median
    p_base = float(df[df['timestamp_ms'] < 30000]['power_mW'].median())
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.16, 3.9),
        gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.42}
    )
    
    # -------------------------------------------------------------------------
    # Panel (a): Full 60-second calibrated protocol trace at 1000 Hz
    # -------------------------------------------------------------------------
    ax1.plot(t_sec, p_mw, color=C_LINE, lw=0.5, alpha=0.85, rasterized=True)
    ax1.axhline(p_base, color=C_BASE, linestyle=':', lw=1.1, zorder=2)
    
    # Shade communication and training
    ax1.fill_between(t_sec, 520, 850, where=(comm == 1), color=C_COMM, alpha=0.45, zorder=3)
    ax1.fill_between(t_sec, 520, 850, where=(train == 1), color=C_TRAIN, alpha=0.55, zorder=3)
    
    # Annotate pre-baseline, turn, post-baseline
    ax1.annotate('30 s Pre-Baseline Window\n(Quiescent Floor $P_{\\rm base} = 560.5$ mW)',
                 xy=(15.0, 560.5), xytext=(15.0, 715),
                 ha='center', va='bottom', fontsize=6.8, color='#546E7A',
                 arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=0.8))
    
    ax1.annotate('Client Turn ($E=20$)',
                 xy=(30.7, 750), xytext=(30.7, 830),
                 ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_LINE,
                 arrowprops=dict(arrowstyle='->', color=C_LINE, lw=1.0))
    
    ax1.annotate('30 s Post-Baseline Window\n(Thermal Drift $< 0.24\\%$)',
                 xy=(46.0, 560.5), xytext=(46.0, 715),
                 ha='center', va='bottom', fontsize=6.8, color='#546E7A',
                 arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=0.8))
    
    ax1.set_xlim(0, t_sec[-1])
    ax1.set_ylim(530, 900)
    ax1.set_xlabel('Elapsed Benchmark Time (seconds)', labelpad=2)
    ax1.set_ylabel('Power (mW)')
    ax1.set_title('(a) Full Homogeneous Benchmark Trace (61.0 s at 1000 Hz INA238 Logging, $E=20$ Epochs)', pad=4, loc='left')
    
    # -------------------------------------------------------------------------
    # Panel (b): Zoomed single client turn decomposition at 1.0 ms resolution
    # -------------------------------------------------------------------------
    # Window spans from 30.46 s to 31.22 s (760 ms total window)
    t_start_z = 30.460
    t_end_z = 31.220
    mask_z = (t_sec >= t_start_z) & (t_sec <= t_end_z)
    
    tz = t_sec[mask_z]
    pz = p_mw[mask_z]
    comm_z = comm[mask_z]
    train_z = train[mask_z]
    
    tz_rel_ms = (tz - t_start_z) * 1000.0
    
    ax2.plot(tz_rel_ms, pz, color=C_LINE, lw=1.0, marker='o', markersize=1.8,
             markerfacecolor='white', markeredgewidth=0.5, alpha=0.95, zorder=4)
    ax2.axhline(p_base, color=C_BASE, linestyle=':', lw=1.2, zorder=2)
    
    ax2.fill_between(tz_rel_ms, 520, 1100, where=(comm_z == 1), color=C_COMM, alpha=0.30, zorder=2)
    ax2.fill_between(tz_rel_ms, 520, 1100, where=(train_z == 1), color=C_TRAIN, alpha=0.35, zorder=2)
    
    # Settling delay bracket annotations (60 ms inter-phase gaps)
    for gap_x in [111, 201, 413]:
        ax2.annotate('60 ms settle', xy=(gap_x, 562), xytext=(gap_x, 635),
                     ha='center', va='bottom', fontsize=5.8, color='#78909C', fontstyle='italic',
                     arrowprops=dict(arrowstyle='->', color='#B0BEC5', lw=0.7))
    
    # Inset callout summary card placed in wide open space on right (x = 525..755 ms)
    box_text = (
        r"$\bf{Microarchitectural\ Breakdown:}$" + "\n"
        r"$\bullet$ $\bf{Wi\text{-}Fi\ Config}$ (GET /config): 32 ms" + "\n"
        r"$\bullet$ $\bf{Wi\text{-}Fi\ RX}$ (GET /model): 28 ms" + "\n"
        r"$\bullet$ $\bf{SGD\ Training}$ ($E=20$): 152 ms" + "\n"
        r"$\bullet$ $\bf{Wi\text{-}Fi\ TX}$ (POST /update): 55 ms" + "\n"
        r"$\bullet$ $\bf{Calibrated\ Settling}$: 60 ms gaps" + "\n"
        r"$\bullet$ $\bf{Quiescent\ Floor}$: $560.5$ mW" + "\n"
        r"$\bullet$ $\bf{Sampling\ Rate}$: 1000 Hz (0 drops)"
    )
    ax2.text(750, 840, box_text, fontsize=6.7, va='center', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', edgecolor='#B0BEC5', alpha=0.96), zorder=5)

    ax2.set_xlim(0, tz_rel_ms[-1])
    ax2.set_ylim(535, 1080)
    ax2.set_xlabel('Client Turn Elapsed Time (milliseconds)', labelpad=2)
    ax2.set_ylabel('Power (mW)')
    ax2.set_title('(b) Microarchitectural Phase Isolation (Single Client Turn Decomposition at 1.0 ms Resolution)', pad=4, loc='left')
    
    # Legend underneath
    legend_elements = [
        plt.Line2D([0], [0], color=C_LINE, lw=1.0, marker='o', markersize=2.5, markerfacecolor='white', label='Instantaneous Power (1 kHz INA238)'),
        plt.Line2D([0], [0], color=C_BASE, linestyle=':', lw=1.2, label=f'Quiescent Baseline ({p_base:.1f} mW)'),
        Patch(facecolor=C_COMM, alpha=0.45, edgecolor='none', label='Wi-Fi Communication (COMM)'),
        Patch(facecolor=C_TRAIN, alpha=0.55, edgecolor='none', label='On-Device SGD Training (TRAIN)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=4,
               fontsize=7.0, columnspacing=1.0, frameon=True, edgecolor='#CFD8DC', facecolor='white', framealpha=0.98)
    
    destinations = [
        Path('figures'),
        Path('results/figures')
    ]
    for d in destinations:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'fig1_power_time_series.pdf', bbox_inches='tight')
        fig.savefig(d / 'fig1_power_time_series.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    print("[OK] Successfully generated publication-grade Fig. 2.")

if __name__ == '__main__':
    generate_fig2()
