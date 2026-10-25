#!/usr/bin/env python3
"""Additional figures for Campaign v2. Usage: python3 -I make_figs_v3.py <data_summary dir> <figure dir>
Reads run_summary.csv, paper_numbers.json, derived.json and analytical_baselines.json. Writes
fig_estar_design.pdf, fig_energy_budget.pdf, fig_model_mismatch.pdf, fig_comm_diagnostics.pdf."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ds, fd = sys.argv[1], sys.argv[2]
S = pd.read_csv(ds + '/run_summary.csv'); N = json.load(open(ds + '/paper_numbers.json')); D = json.load(open(ds + '/derived.json'))
AB = json.load(open(ds + '/analytical_baselines.json'))
BL, OR, GR, RD, AM, PU = '#0072B2', '#D55E00', '#6b6b6b', '#009E73', '#E69F00', '#CC79A7'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'font.family': 'serif', 'pdf.fonttype': 42})
a, c0, dEc, Pb, tep = D['a'], D['c0'], D['dEc'], D['Pb'], D['tep']
M, T = S[S.campaign_step == 'step1_main'], S[S.campaign_step == 'step2_tail']
Tm = N['Tcomm']['median']

# ---------- Fig crossover against exchange time (one column, model with assumed sleep values)
def estar(Ps, Tc_ms, kc=1.0):
    dP = (Pb - Ps) / 1000.0  # W -> mJ per ms
    return (dP * Tc_ms + kc * dEc - c0) / (dP * tep + a)
sl = {s['state']: s for s in D['sleep']}
fig, ax = plt.subplots(figsize=(3.4, 2.5))
tc = np.linspace(50, 300, 200)
ax.axvspan(N['Tcomm']['q25'], N['Tcomm']['q75'], color='#eeeeee', lw=0, zorder=0)
ax.text((N['Tcomm']['q25'] + N['Tcomm']['q75']) / 2, 1.0, 'middle half\nof 105 runs', ha='center', va='bottom', fontsize=6.5, color=GR)
for st, col, ls, lab in [('Standby', GR, '--', 'Standby (no sleep)'), ('Modem sleep', PU, '-', r'Modem sleep, $P_s=100$ mW'),
                         ('Deep sleep', RD, '-', r'Deep sleep, $P_s=0.08$ mW')]:
    ax.plot(tc, estar(sl[st]['Ps'], tc), color=col, ls=ls, lw=1.3, label=lab)
ax.axvline(Tm, color=GR, lw=0.7, ls=':')
ax.text(Tm + 3, 27.5, 'median', fontsize=6.5, color=GR)
ax.set_xlim(50, 300); ax.set_ylim(0, 32)
ax.set_xlabel(r'Exchange time $T_{\rm comm}$ (ms)'); ax.set_ylabel(r'Crossover $E^\star$ (epochs)')
ax.legend(frameon=False, loc='lower right', fontsize=6.5)
plt.tight_layout(); plt.savefig(fd + '/fig_estar_design.pdf'); plt.close()

# ---------- Fig energy budget per turn (measured fits, T_comm = median)
Es = [1, 2, 5, 8, 10, 20]; x = np.arange(len(Es))
base = np.array([Pb * (Tm + tep * e) / 1000.0 for e in Es]); comp = np.array([a * e + c0 for e in Es]); comm = np.full(len(Es), dEc)
fig, ax = plt.subplots(figsize=(3.4, 2.4)); w = 0.62; kw = dict(width=w, edgecolor='white', linewidth=0.8)
ax.bar(x, base, color='#c4c4c4', label=r'Awake baseline $P_{\rm base}T_{\rm active}$', **kw)
ax.bar(x, comm, bottom=base, color=OR, label=r'Marginal comm. $\Delta E_{\rm comm}$', **kw)
ax.bar(x, comp, bottom=base + comm, color=BL, label=r'Marginal comp. $\Delta E_{\rm comp}$', **kw)
for i in range(len(Es)):
    tot = base[i] + comm[i] + comp[i]
    ax.text(x[i], tot + 3, f'{100 * base[i] / tot:.0f}%', ha='center', va='bottom', fontsize=6.5, color='#333333')
ax.set_xticks(x); ax.set_xticklabels([str(e) for e in Es]); ax.set_xlabel('Local epochs $E$'); ax.set_ylabel('Energy per client turn (mJ)')
ax.set_ylim(0, 215); ax.grid(axis='x', visible=False); ax.legend(frameon=False, loc='upper left', fontsize=6.5)
plt.tight_layout(); plt.savefig(fd + '/fig_energy_budget.pdf'); plt.close()

# ---------- Fig analytical models vs measurement (log axes). Model values from analytical_baselines.json.
# Rows are the same in every panel; the top row is the analytical model of that panel, named next to its marker.
# (label, nominal, [lo, hi] or None, kind, note) kind: 'model' | 'meas' | 'meas2'
ROWS = ['Analytical model', 'Datasheet currents', 'Measured, main', 'Measured, 2nd session']
nlh = lambda k: (AB[k]['nom'], [AB[k]['lo'], AB[k]['hi']])
panels = [(r'(a) $\Delta E_{\rm comm}$ (mJ per turn)', (0.005, 100), [
              ('Shannon capacity', *nlh('shannon'), 'model', ''), ('Datasheet currents', *nlh('ds_comm'), 'model', ''),
              ('Measured, main', dEc, N['comm_mean_ci'], 'meas', ''), ('Measured, 2nd session', N['step2_60']['comm'], None, 'meas2', '')]),
          (r'(b) $\Delta E_{\rm comp}$ (mJ per epoch)', (0.005, 10), [
              ('Switching energy', *nlh('switching'), 'model', ''), ('Datasheet currents', *nlh('ds_comp'), 'model', 'calibrated'),
              ('Measured, main', a, [0.991, 1.014], 'meas', ''), ('Measured, 2nd session', N['step2_60']['a'], None, 'meas2', '')]),
          (r'(c) $E^\star$ (epochs)', (0.05, 100), [
              ('Shannon + switching', *nlh('combined'), 'model', ''), ('Datasheet currents', *nlh('ds_Estar'), 'model', ''),
              ('Measured, main', N['Estar_A'], N['Estar_A_ci'], 'meas', ''),
              ('Measured, 2nd session', N['step2_60']['Estar'], N['step2_60']['Estar_ci'], 'meas2', '')])]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=False)
for k, (title, xl, rws) in enumerate(panels):
    A = ax[k]; A.set_xscale('log'); A.set_xlim(*xl); A.set_title(title, loc='left')
    A.axvspan(rws[2][2][0], rws[2][2][1], color='#dbe9f5', lw=0, zorder=0)
    for i, (lab, v, ci, kind, note) in enumerate(rws):
        y = len(rws) - 1 - i; col = {'model': GR, 'meas': BL, 'meas2': AM}[kind]
        if ci: A.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], color=col, lw=1.1, capsize=2, fmt='none')
        A.plot(v, y, marker='D' if kind == 'model' else 'o', ms=4.5, color=col, mfc='white' if kind == 'model' else col, mew=1.1, zorder=3)
        if i == 0:
            A.text(v, y + 0.28, lab, ha='center', va='bottom', fontsize=6.5, color='#333333')
        if note:
            A.text(ci[1] * 1.3, y, note, ha='left', va='center', fontsize=6.5, color=GR, style='italic')
    A.set_yticks(range(len(rws))); A.grid(axis='y', visible=False)
    A.set_yticklabels(ROWS[::-1] if k == 0 else [])
    if k > 0: A.tick_params(axis='y', length=0)
    A.set_ylim(-0.6, len(rws) - 0.05)
plt.tight_layout(w_pad=0.6); plt.savefig(fd + '/fig_model_mismatch.pdf'); plt.close()

# ---------- Fig communication diagnostics: exchange-time distribution and run-order drift
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax[0].axvspan(N['Tcomm']['q25'], N['Tcomm']['q75'], color='#eeeeee', lw=0, zorder=0)
for g, col, lab in [(M, BL, 'Main campaign (60)'), (T, AM, 'Second session (45)'), (S, 'k', 'All runs (105)')]:
    v = np.sort(g.T_comm_ms.values); y = np.arange(1, len(v) + 1) / len(v)
    ax[0].step(v, y, where='post', color=col, lw=1.0 if col == 'k' else 1.3, ls='-' if col != 'k' else (0, (1, 1)), label=lab)
ax[0].axvline(500, color=OR, lw=0.8, ls='--'); ax[0].text(540, 0.62, f"{N['Tcomm']['n_gt500']} runs\n> 500 ms", fontsize=6.5, color=OR, va='center')
ax[0].set_xscale('log'); ax[0].set_xlim(40, 4000); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel(r'Download + upload time $T_{\rm comm}$ (ms)'); ax[0].set_ylabel('Fraction of runs')
ax[0].legend(frameon=False, loc='lower right'); ax[0].set_title('(a) Exchange time', loc='left')
for g, col, lab in [(M, BL, 'Main campaign'), (T, AM, 'Second session')]:
    g = g.sort_values('order_index'); xo = g.order_index.values; e = g.marginal_comm_energy_mJ.values
    ax[1].scatter(xo, e, s=9, color=col, alpha=.75, lw=0, label=lab)
    for chunk in np.array_split(np.arange(len(g)), 3):
        ax[1].plot([xo[chunk[0]] - 0.5, xo[chunk[-1]] + 0.5], [e[chunk].mean()] * 2, color=col, lw=1.6)
        ax[1].text(xo[chunk].mean(), 22.4, f'{e[chunk].mean():.1f}', ha='center', fontsize=6.5, color=col)
ax[1].axvline(60.5, color=GR, lw=0.6)
ax[1].set_ylim(0, 27); ax[1].set_xlim(0, 106)
ax[1].text(30.5, 24.6, 'Main (randomized)', ha='center', fontsize=6.5, color=BL)
ax[1].text(83, 24.6, 'Second (blocked)', ha='center', fontsize=6.5, color=AM)
outl = S[S.marginal_comm_energy_mJ > 25]
for _, r in outl.iterrows():  # off-scale point, drawn at the top edge
    c = BL if r.campaign_step == 'step1_main' else AM
    ax[1].plot(r.order_index, 26.4, marker='^', ms=4, color=c, clip_on=False)
    ax[1].text(r.order_index + 1.5, 26.4, f'{r.marginal_comm_energy_mJ:.1f}', va='center', fontsize=6, color=GR)
ax[1].set_xlabel('Acquisition order'); ax[1].set_ylabel(r'$\Delta E_{\rm comm}$ per turn (mJ)')
ax[1].set_title('(b) Communication energy by run order', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_comm_diagnostics.pdf'); plt.close()
print('ok')
