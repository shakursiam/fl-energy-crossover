#!/usr/bin/env python3
"""Figures from Campaign v2. Usage: python3 -I make_figs_v2.py <raw_v2 dir> <analysis_out dir> <figure dir>"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
root, ao, fd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(fd, exist_ok=True)
S = pd.read_csv(root + '/run_summary.csv'); N = json.load(open(ao + '/paper_numbers.json')); D = json.load(open(ao + '/derived.json'))
ST = json.load(open(ao + '/stats.json')); TP = json.load(open(ao + '/tail_profile.json'))
BL, OR, GR, RD = '#0072B2', '#D55E00', '#6b6b6b', '#009E73'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'font.family': 'serif', 'pdf.fonttype': 42})
M = S[S.campaign_step == 'step1_main']; a, c0, dEc = N['fit']['a'], N['fit']['c'], N['comm_mean']
rng = np.random.default_rng(1)
# ---------- Fig crossover
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for e, g in M.groupby('E'):
    j = rng.uniform(-0.25, 0.25, len(g)) * (1 + 0.15*e)
    ax[0].scatter(e+j, g.marginal_comp_energy_mJ, s=7, color=BL, alpha=.5, lw=0)
    ax[0].scatter(e+j, g.marginal_comm_energy_mJ, s=7, color=OR, alpha=.5, lw=0, marker='s')
pe = pd.DataFrame(N['perE'])
ax[0].plot(pe.E, pe.comm, color=OR, marker='s', ms=4, lw=1.2, label=r'Communication $\Delta E_{\rm comm}$ (mean)')
x = np.linspace(0, 21, 50); ax[0].plot(x, a*x+c0, color=BL, lw=1.2, label=r'Computation fit $%.3fE%+.3f$' % (a, c0))
ax[0].scatter(pe.E, pe.comp, s=14, color=BL, zorder=3)
ax[0].axvline(N['Estar_A'], color=GR, ls='--', lw=0.9); ax[0].text(N['Estar_A']+0.3, 1.0, r'$E^\star=%.1f$' % N['Estar_A'], color=GR, fontsize=7)
ax[0].set_xlabel('Local epochs $E$'); ax[0].set_ylabel('Marginal energy per turn (mJ)'); ax[0].legend(frameon=False, loc='upper left'); ax[0].set_title('(a) Energy per turn, 60 runs', loc='left')
ratio = M.groupby('E').apply(lambda g: g.marginal_comp_energy_mJ.mean()/g.marginal_comm_energy_mJ.mean())
ax[1].plot(ratio.index, ratio.values, color=BL, marker='o', ms=4, lw=1.2); ax[1].axhline(1, color=GR, ls='--', lw=0.9)
ax[1].set_xlabel('Local epochs $E$'); ax[1].set_ylabel('Computation / communication'); ax[1].set_title('(b) Ratio of mean energies', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_crossover_v2.pdf'); plt.savefig(fd + '/fig_crossover_v2.png', dpi=300); plt.close()
# ---------- Fig tail A/B
T = S[S.campaign_step == 'step2_tail']; fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
cols = {0: OR, 60: BL, 200: RD}; mk = {0: 'o', 60: 's', 200: '^'}
for d, g in T.groupby('delay_ms'):
    m = g.groupby('E').marginal_comp_energy_mJ.mean(); sd = g.groupby('E').marginal_comp_energy_mJ.std()
    resid = g.marginal_comp_energy_mJ - a*g.E
    mm = resid.groupby(g.E).mean(); ss = resid.groupby(g.E).std()
    ax[0].errorbar(mm.index + {0: -0.5, 60: 0, 200: 0.5}[d], mm.values, yerr=ss.values, color=cols[d], marker=mk[d], ms=4, lw=1, capsize=2, label=f'{d} ms')
ax[0].axhline(0, color=GR, lw=.7); ax[0].set_xlabel('Local epochs $E$'); ax[0].set_ylabel(r'Training energy minus $\alpha E$ (mJ)'); ax[0].legend(title='Settling delay', frameon=False); ax[0].set_title('(a) Excess training energy', loc='left')
tl = ST['tail_ab']; xs = [0, 60, 200]
for i, d in enumerate(xs):
    lo, hi = tl[str(d)]['intercept_ci']; v = tl[str(d)]['intercept']
    ax[1].errorbar(i, v, yerr=[[v-lo], [hi-v]], color=cols[d], marker=mk[d], ms=5, capsize=3, lw=1.2)
ax[1].axhline(0, color=GR, lw=.7); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['0 ms', '60 ms', '200 ms']); ax[1].set_xlim(-.5, 2.5)
ax[1].set_ylabel(r'Fitted intercept $c_0$ (mJ), 95% CI'); ax[1].set_xlabel('Settling delay'); ax[1].set_title('(b) Intercept by delay', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_tail_ab.pdf'); plt.savefig(fd + '/fig_tail_ab.png', dpi=300); plt.close()
# ---------- Fig tail profile
fig, ax = plt.subplots(figsize=(3.4, 2.5)); t = np.arange(60)
for k, lab, col, ls in [('after_dl', 'After download', BL, '-'), ('after_cfg', 'After config request', RD, '-'), ('after_ul', 'After upload', OR, '-'), ('after_train', 'After training', GR, '--')]:
    y = np.array(TP[k]['mean_excess_mW_ms'])[:60]; ax.plot(t, y, color=col, ls=ls, lw=1.1, label=f"{lab} ({TP[k]['energy_first_60ms_mJ']:.2f} mJ)")
ax.set_xlabel('Time after the end of the phase (ms)'); ax.set_ylabel('Mean power above baseline (mW)'); ax.legend(frameon=False, fontsize=6.5)
plt.tight_layout(); plt.savefig(fd + '/fig_tail_profile.pdf'); plt.savefig(fd + '/fig_tail_profile.png', dpi=300); plt.close()
# ---------- Fig variability
fig, ax = plt.subplots(figsize=(3.4, 2.6)); 
groups = [('Main\nE=1', M[M.E == 1]), ('Main\nE=5', M[M.E == 5]), ('Main\nE=20', M[M.E == 20]), ('Second\nsession', T[T.delay_ms == 60])]
for i, (lab, g) in enumerate(groups):
    y = g.marginal_comm_energy_mJ.values; j = rng.uniform(-.18, .18, len(y)); ax.scatter(i+j, y, s=9, color=BL if i < 3 else OR, alpha=.7, lw=0)
    ax.plot([i-.3, i+.3], [y.mean()]*2, color='k', lw=1.2)
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=6.5); ax.set_ylabel(r'$\Delta E_{\rm comm}$ per turn (mJ)'); plt.tight_layout(); plt.savefig(fd + '/fig_comm_variation.pdf'); plt.savefig(fd + '/fig_comm_variation.png', dpi=300); plt.close()
# ---------- Fig convergence energy
cv = pd.DataFrame(D['conv']); fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax[0].plot(cv.E, cv.R, color=BL, marker='o', ms=4); ax[0].set_xlabel('Local epochs $E$'); ax[0].set_ylabel('Rounds to 75% accuracy'); ax[0].set_title('(a) Simulated rounds', loc='left')
for k, lab, col, m_ in [('marg', 'Marginal', BL, 'o'), ('gross', 'Gross awake', OR, 's'), ('deep', 'Deep sleep (assumed)', RD, '^')]:
    ax[1].plot(cv.E, cv[k]/cv[k].max(), color=col, marker=m_, ms=4, label=lab)
ax[1].set_xlabel('Local epochs $E$'); ax[1].set_ylabel('Energy to target / its maximum'); ax[1].legend(frameon=False); ax[1].set_title('(b) Energy to target', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_energy_to_target_v2.pdf'); plt.savefig(fd + '/fig_energy_to_target_v2.png', dpi=300); plt.close()
print('ok')
