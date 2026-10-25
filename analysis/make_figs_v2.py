#!/usr/bin/env python3
"""Figures from Campaign v2. Usage: python3 -I make_figs_v2.py <raw_v2 dir> <analysis_out dir> <figure dir>"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
root, ao, fd = sys.argv[1], sys.argv[2], sys.argv[3]
S = pd.read_csv(root + '/run_summary.csv'); N = json.load(open(ao + '/paper_numbers.json')); D = json.load(open(ao + '/derived.json'))
ST = json.load(open(ao + '/stats.json')); TP = json.load(open(ao + '/tail_profile.json'))
BL, OR, GR, RD = '#0072B2', '#D55E00', '#6b6b6b', '#009E73'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'font.family': 'serif', 'pdf.fonttype': 42})
M = S[S.campaign_step == 'step1_main']; a, c0, dEc = N['fit']['a'], N['fit']['c'], N['comm_mean']
rng = np.random.default_rng(1)
# ---------- Fig crossover (full text width)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
for e, g in M.groupby('E'):
    j = rng.uniform(-0.25, 0.25, len(g)) * (1 + 0.15*e)
    ax[0].scatter(e+j, g.marginal_comp_energy_mJ, s=8, color=BL, alpha=.5, lw=0)
    ax[0].scatter(e+j, g.marginal_comm_energy_mJ, s=8, color=OR, alpha=.5, lw=0, marker='s')
pe = pd.DataFrame(N['perE'])
# communication is modelled as constant in E: pooled mean of the 60 runs with its 95% bootstrap interval
cm_ci = N['comm_mean_ci']
x = np.linspace(0, 21, 50)
ax[0].axhspan(*cm_ci, color=OR, alpha=0.15, lw=0, zorder=0)
ax[0].axhline(dEc, color=OR, lw=1.3, label=r'Communication, pooled mean %.2f mJ (95%% CI)' % dEc)
ax[0].scatter(pe.E, pe.comm, s=16, facecolor='white', edgecolor=OR, marker='s', lw=1.0, zorder=3, label='Communication, mean per $E$')
ax[0].plot(x, a*x+c0, color=BL, lw=1.3, label=r'Computation fit $%.3fE%+.3f$ mJ' % (a, c0))
ax[0].scatter(pe.E, pe.comp, s=16, color=BL, zorder=3)
ax[0].axvspan(*N['Estar_A_ci'], color='#e6e6e6', lw=0, zorder=0)
ax[0].axvline(N['Estar_A'], color=GR, ls='--', lw=0.9); ax[0].text(N['Estar_A_ci'][1]+0.3, 1.0, r'$E^\star=%.1f$ [%.1f, %.1f]' % (N['Estar_A'], *N['Estar_A_ci']), color=GR, fontsize=7)
ax[0].set_ylim(-1, 31); ax[0].set_xlabel('Local epochs $E$'); ax[0].set_ylabel('Marginal energy per turn (mJ)'); ax[0].legend(frameon=False, loc='upper left', fontsize=6.5)
ax[0].set_title('(a) Energy per turn, 60 runs', loc='left')
ratio = M.groupby('E').apply(lambda g: g.marginal_comp_energy_mJ.mean()/g.marginal_comm_energy_mJ.mean())
bs = np.array([[rng.choice(g.marginal_comp_energy_mJ.values, len(g)).mean() / rng.choice(g.marginal_comm_energy_mJ.values, len(g)).mean()
                for _ in range(4000)] for e, g in M.groupby('E')])
lo, hi = np.percentile(bs, 2.5, axis=1), np.percentile(bs, 97.5, axis=1)
ax[1].errorbar(ratio.index, ratio.values, yerr=[ratio.values - lo, hi - ratio.values], color=BL, marker='o', ms=4, lw=1.2, capsize=2.5)
ax[1].axhline(1, color=GR, ls='--', lw=0.9)
ax[1].set_xlabel('Local epochs $E$'); ax[1].set_ylabel('Computation / communication'); ax[1].set_title('(b) Ratio of mean energies, 95% bootstrap CI', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_crossover_v2.pdf'); plt.close()
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
    sl, (slo, shi) = tl[str(d)]['slope'], tl[str(d)]['slope_ci']
    ax[1].text(i, -0.42, '$\\alpha=%.3f$\n[%.3f, %.3f]\n$R^2=%.3f$' % (sl, slo, shi, tl[str(d)]['r2']), ha='center', va='top', fontsize=6.5, color='#333333')
ax[1].axhline(0, color=GR, lw=.7); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['0 ms', '60 ms', '200 ms']); ax[1].set_xlim(-.5, 2.5); ax[1].set_ylim(-0.95, 0.95)
ax[1].set_ylabel(r'Fitted intercept $c_0$ (mJ), 95% CI'); ax[1].set_xlabel('Settling delay'); ax[1].set_title('(b) Fitted intercept and slope by delay', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_tail_ab.pdf'); plt.close()
# ---------- Fig tail profile
fig, ax = plt.subplots(figsize=(3.4, 2.5)); t = np.arange(60)
for k, lab, col, ls in [('after_dl', 'After download', BL, '-'), ('after_cfg', 'After config request', RD, '-'), ('after_ul', 'After upload', OR, '-'), ('after_train', 'After training', GR, '--')]:
    y = np.array(TP[k]['mean_excess_mW_ms'])[:60]; ax.plot(t, y, color=col, ls=ls, lw=1.1, label=f"{lab} ({TP[k]['energy_first_60ms_mJ']:.2f} mJ)")
ax.set_xlabel('Time after the end of the phase (ms)'); ax.set_ylabel('Mean power above baseline (mW)'); ax.legend(frameon=False, fontsize=6.5)
plt.tight_layout(); plt.savefig(fd + '/fig_tail_profile.pdf'); plt.close()
print('ok')
