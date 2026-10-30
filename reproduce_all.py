#!/usr/bin/env python3
"""Regenerates every number and figure of the paper from the raw traces.
Usage (from the repository root): python reproduce_all.py [--convergence]
--convergence reruns the FL simulation of Section VIII (several minutes); by default its figure
is redrawn from the saved results in data_summary/convergence_curves.npz, with the energies recomputed
from the current derived.json.
Outputs: data_summary/*.json and figures/*.pdf."""
import sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW, DS, FIG, AN = 'data/raw_v2', 'data_summary', 'figures', 'analysis'
SLEEP = 'data/raw_sleep/session_01'
FEAT = 'data/processed/kws_disjoint_harder_features.npz'
(ROOT / FIG).mkdir(exist_ok=True)

def run(*args):
    cmd = [sys.executable, '-I', f'{AN}/{args[0]}', *args[1:]]
    print('>>', ' '.join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)

run('verify_campaign_v2.py', RAW, DS)              # trace checks, recomputed_runs.csv
run('tail_profile_v2.py', RAW, DS)                 # tail_profile.json
run('stats_campaign_v2.py', DS)                    # stats.json
run('paper_numbers_v2.py', DS, DS)                 # fits, crossover estimates and intervals
run('analytical_baselines.py', DS)                 # Section IV models
run('analyze_sleep_campaign.py', SLEEP, '--out', f'{DS}/sleep_campaign.json')   # measured sleep power, wake-up energy
run('derived_v2.py', DS)                           # sleep, contention, model mismatch
run('convergence_v3.py', FEAT, DS, FIG, *([] if '--convergence' in sys.argv else ['reprice']))
run('convergence_robustness.py', DS)               # targets 75-78%, wake-up energy
run('make_figs_v2.py', DS, DS, FIG)
run('make_figs_v3.py', DS, FIG)
run('plot_fig2_new_logger.py', FIG)
run('make_fig_sleep_trace.py', SLEEP, DS, FIG)     # measured sleep cycle and wake-up
run('make_fig_wiring.py')                           # testbed wiring

N = json.load(open(ROOT / DS / 'paper_numbers.json'))
print(f"\nalpha = {N['fit']['a']:.3f} mJ/epoch, c0 = {N['fit']['c']:.3f} mJ, dE_comm = {N['comm_mean']:.2f} mJ")
print(f"E* = {N['Estar_A']:.2f} [{N['Estar_A_ci'][0]:.2f}, {N['Estar_A_ci'][1]:.2f}] (main), "
      f"{N['step2_60']['Estar']:.2f} (second session), with tail {N['Estar_B']:.2f}")
S = json.load(open(ROOT / DS / 'sleep_campaign.json'))['summary']
print(f"sleep: modem {S['P_modem_mW']['mean']:.1f} mW, light {S['P_light_mW']['mean']:.1f} mW, deep {S['P_deep_mW']['mean']:.2f} mW, "
      f"E_wake deep {S['E_wake_deep_mJ']['mean']:.0f} mJ, light {S['E_wake_light_mJ']['mean']:.0f} mJ")
