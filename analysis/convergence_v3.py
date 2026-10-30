#!/usr/bin/env python3
"""FL convergence simulation for Section VIII, regenerated with the Campaign v2 energy fits.
Usage (from the repository root):
  python3 -I convergence_v3.py <features.npz> <data_summary dir> <figure dir>

Setup: K=20 clients, 100 disjoint clips each, Dirichlet label skew (concentration 0.3), batch 16,
FedAvg on the 780-parameter linear head, 20 seeds (42-61), E in {1,2,5,8,10,20}, up to 30 rounds.
The test pool is split per class into a validation half (learning-rate selection only) and a test
half (all reported numbers). Rounds to 75% test accuracy are interpolated linearly between rounds.
Energies combine R(E) with the measured fits in derived.json (T_comm = median, 125 ms).
Writes <data_summary>/convergence.json, convergence_curves.npz and <figure dir>/fig_convergence_v3.pdf.
Add a 4th argument 'plot' to redraw the figure from the saved results, or 'reprice' to recompute the
energies and optima from the saved per-seed rounds with the current derived.json (no retraining).
"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

feat, ds, fd = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else 'run'
PLOT_ONLY = MODE in ('plot', 'reprice')   # reuse saved results
K, N_K, B, ROUNDS, ALPHA_DIR = 20, 100, 16, 30, 0.3
SEEDS = list(range(42, 62))
EPOCHS = [1, 2, 5, 8, 10, 20]
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TARGET = 75.0
TR_S = 60.0
_deep = [s for s in json.load(open(ds + '/derived.json'))['sleep'] if s['state'] == 'Deep sleep'][0]
PS_DEEP_MW, EWAKE_DEEP_MJ = _deep['Ps'], _deep['wake']   # Campaign 4 measured values (assumed: 0.08 mW, 180 mJ)
DEEP_SOURCE = json.load(open(ds + '/derived.json')).get('sleep_source', 'assumed')

if not PLOT_ONLY:
    d = np.load(feat)
    xtr, ytr, xte_all, yte_all = d['x_train'], d['y_train'], d['x_test'], d['y_test']
    C = len(d['labels'])
    # per-class split of the test pool into validation and test halves (fixed)
    rs = np.random.RandomState(0); val_idx, test_idx = [], []
    for c in np.unique(yte_all):
        idx = np.where(yte_all == c)[0]; rs.shuffle(idx); h = len(idx) // 2
        val_idx += idx[:h].tolist(); test_idx += idx[h:].tolist()
    xva, yva, xte, yte = xte_all[val_idx], yte_all[val_idx], xte_all[test_idx], yte_all[test_idx]

def partition(seed):
    """Dirichlet label skew, then exactly N_K disjoint clips per client."""
    rng = np.random.RandomState(seed)
    classes = np.unique(ytr); by_c = {c: rng.permutation(np.where(ytr == c)[0]).tolist() for c in classes}
    props = rng.dirichlet(np.repeat(ALPHA_DIR, K), size=len(classes))
    pref = [[] for _ in range(K)]
    for ci, c in enumerate(classes):
        cnt = (props[ci] * len(by_c[c])).astype(int); cur = 0
        for k in range(K):
            pref[k] += by_c[c][cur:cur + cnt[k]]; cur += cnt[k]
    used, parts = set(), []
    pool = rng.permutation(len(ytr)).tolist()
    for k in range(K):
        mine = [i for i in pref[k] if i not in used][:N_K]
        if len(mine) < N_K:
            extra = [i for i in pool if i not in used and i not in mine][:N_K - len(mine)]
            mine += extra
        used.update(mine); parts.append(np.array(mine))
    assert len(used) == K * N_K
    return parts

def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def acc(w, b, x, y):
    return float(np.mean(np.argmax(x @ w + b, axis=1) == y) * 100)

def local_sgd(w, b, x, y, epochs, lr, rng):
    w, b = w.copy(), b.copy(); n = len(y)
    for _ in range(epochs):
        p = rng.permutation(n)
        for s in range(0, n, B):
            ib = p[s:s + B]; xb, yb = x[ib], y[ib]
            g = softmax(np.clip(xb @ w + b, -50, 50)); g[np.arange(len(yb)), yb] -= 1
            w -= lr * (xb.T @ g) / len(yb); b -= lr * g.mean(axis=0)
    return w, b

def run(seed, e, lr, parts):
    rng = np.random.default_rng(seed * 100 + e)
    w = np.random.RandomState(seed).normal(0, 0.05, (xtr.shape[1], C)); b = np.zeros(C)
    hv, ht = [acc(w, b, xva, yva)], [acc(w, b, xte, yte)]
    for _ in range(ROUNDS):
        ups = [local_sgd(w, b, xtr[ix], ytr[ix], e, lr, rng) for ix in parts]
        w = np.mean([u[0] for u in ups], axis=0); b = np.mean([u[1] for u in ups], axis=0)
        if not np.isfinite(w).all():
            return None
        hv.append(acc(w, b, xva, yva)); ht.append(acc(w, b, xte, yte))
    return np.array(hv), np.array(ht)

def rounds_to(h, target):
    """Fractional rounds; h[0] is the initial model. NaN if never reached."""
    for r in range(1, len(h)):
        if h[r] >= target:
            return r - 1 + (target - h[r - 1]) / (h[r] - h[r - 1]) if h[r] > h[r - 1] else float(r)
    return np.nan

def boot(v, n=4000):
    v = np.asarray(v, float); r = np.random.RandomState(1)
    m = [r.choice(v, len(v)).mean() for _ in range(n)]; return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

D = json.load(open(ds + '/derived.json'))
a, c0, dEc, Pb, tep = D['a'], D['c0'], D['dEc'], D['Pb'], D['tep']; Tc = D['Tcomm_med']
def turn_mj(E):
    marg = dEc + a * E + c0; tact = Tc + tep * E
    gross = Pb * tact / 1000 + marg
    deep = gross + PS_DEEP_MW * (TR_S * 1000 - tact) / 1000 + EWAKE_DEEP_MJ
    return marg, gross, deep

if not PLOT_ONLY:
    parts = {s: partition(s) for s in SEEDS}
    out = {'setup': dict(K=K, clips_per_client=N_K, batch=B, rounds=ROUNDS, dirichlet=ALPHA_DIR, seeds=SEEDS,
                         target_test_acc=TARGET, val_clips=len(yva), test_clips=len(yte),
                         classes_present=[str(d['labels'][c]) for c in np.unique(ytr)]), 'perE': {}}
    curves = {}
    for e in EPOCHS:
        best = None
        for lr in LRS:
            res = [run(s, e, lr, parts[s]) for s in SEEDS]
            if any(r is None for r in res):
                continue
            rv = [rounds_to(r[0], TARGET) for r in res]
            if np.isnan(rv).any():
                continue
            score = float(np.mean(rv))
            if best is None or score < best[0]:
                best = (score, lr, res)
        _, lr, res = best
        R = np.array([rounds_to(r[1], TARGET) for r in res])
        ok = ~np.isnan(R)
        m, g, dp = turn_mj(e)
        out['perE'][str(e)] = dict(lr=lr, R_mean=float(np.nanmean(R)), R_ci=boot(R[ok]), reached=int(ok.sum()),
                                   final_acc=float(np.mean([r[1][-1] for r in res])),
                                   R_seeds=[None if np.isnan(x) else float(x) for x in R],
                                   E_marg_J=float(np.nanmean(R) * m / 1000), E_gross_J=float(np.nanmean(R) * g / 1000),
                                   E_deep_J=float(np.nanmean(R) * dp / 1000),
                                   E_gross_ci=[x * g / 1000 for x in boot(R[ok])])
        curves[e] = np.array([r[1] for r in res])
        print(e, lr, out['perE'][str(e)]['R_mean'], out['perE'][str(e)]['R_ci'], ok.sum(), flush=True)

    # per-seed optimum for each accounting
    for key, idx in [('marg', 0), ('gross', 1), ('deep', 2)]:
        best_e = []
        for si in range(len(SEEDS)):
            cost = {e: out['perE'][str(e)]['R_seeds'][si] * turn_mj(e)[idx] for e in EPOCHS
                    if out['perE'][str(e)]['R_seeds'][si] is not None}
            best_e.append(min(cost, key=cost.get))
        out['opt_' + key] = {'mean_based': min(EPOCHS, key=lambda e: out['perE'][str(e)]['E_' + {'marg': 'marg', 'gross': 'gross', 'deep': 'deep'}[key] + '_J']),
                             'per_seed_counts': {str(e): best_e.count(e) for e in EPOCHS}}
    json.dump(out, open(ds + '/convergence.json', 'w'), indent=1)
    np.savez_compressed(ds + '/convergence_curves.npz', **{f'E{e}': curves[e] for e in EPOCHS})
else:
    out = json.load(open(ds + '/convergence.json'))
    z = np.load(ds + '/convergence_curves.npz'); curves = {e: z[f'E{e}'] for e in EPOCHS}
    if MODE == 'reprice':
        for e in EPOCHS:
            p = out['perE'][str(e)]; m, g, dp = turn_mj(e)
            p.update(E_marg_J=p['R_mean'] * m / 1000, E_gross_J=p['R_mean'] * g / 1000, E_deep_J=p['R_mean'] * dp / 1000,
                     E_gross_ci=[x * g / 1000 for x in p['R_ci']])
        for key, idx in [('marg', 0), ('gross', 1), ('deep', 2)]:
            best_e = []
            for si in range(len(SEEDS)):
                cost = {e: out['perE'][str(e)]['R_seeds'][si] * turn_mj(e)[idx] for e in EPOCHS
                        if out['perE'][str(e)]['R_seeds'][si] is not None}
                best_e.append(min(cost, key=cost.get))
            out['opt_' + key] = {'mean_based': min(EPOCHS, key=lambda e: out['perE'][str(e)]['E_' + key + '_J']),
                                 'per_seed_counts': {str(e): best_e.count(e) for e in EPOCHS}}
        out['deep_sleep_inputs'] = dict(Ps_mW=PS_DEEP_MW, Ewake_mJ=EWAKE_DEEP_MJ, Tr_s=TR_S, source=DEEP_SOURCE)
        json.dump(out, open(ds + '/convergence.json', 'w'), indent=1)
        print(json.dumps({e: {k: out['perE'][str(e)][k] for k in ('E_marg_J', 'E_gross_J', 'E_deep_J')} for e in EPOCHS}, indent=1))
        print({k: out[k] for k in ('opt_marg', 'opt_gross', 'opt_deep')})

# ---------- figure
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'font.family': 'serif', 'pdf.fonttype': 42})
cols = {1: '#0072B2', 2: '#56B4E9', 5: '#009E73', 8: '#E69F00', 10: '#D55E00', 20: '#CC79A7'}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
r = np.arange(ROUNDS + 1)
for e in EPOCHS:
    c = curves[e]; mu = c.mean(0); lo, hi = np.percentile(c, 2.5, 0), np.percentile(c, 97.5, 0)
    ax[0].fill_between(r, lo, hi, color=cols[e], alpha=0.15, lw=0); ax[0].plot(r, mu, color=cols[e], lw=1.2, label=f'$E={e}$')
ax[0].axhline(TARGET, color='#6b6b6b', ls='--', lw=0.8); ax[0].text(11.8, TARGET - 3.5, '75% target', ha='right', fontsize=6.5, color='#6b6b6b')
ax[0].set_xlim(0, 12); ax[0].set_ylim(30, 85); ax[0].set_xlabel('Round'); ax[0].set_ylabel('Test accuracy (%)')
ax[0].legend(frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(1.0, 0.08)); ax[0].set_title('(a) Test accuracy, 20 seeds', loc='left')
for key, lab, col, mk in [('E_marg_J', 'Marginal', '#0072B2', 'o'), ('E_gross_J', 'Gross awake', '#D55E00', 's'), ('E_deep_J', 'Deep sleep (' + DEEP_SOURCE + ')', '#009E73', '^')]:
    v = np.array([out['perE'][str(e)][key] for e in EPOCHS])
    ci = np.array([[x * v[i] / out['perE'][str(EPOCHS[i])]['R_mean'] for x in out['perE'][str(EPOCHS[i])]['R_ci']] for i in range(len(EPOCHS))])
    vmax = v.max()
    ax[1].errorbar(EPOCHS, v / vmax, yerr=[(v - ci[:, 0]) / vmax, (ci[:, 1] - v) / vmax], color=col, marker=mk, ms=4, lw=1.1, capsize=2, label=lab)
ax[1].set_xlabel('Local epochs $E$'); ax[1].set_ylabel('Energy to target / its maximum'); ax[1].legend(frameon=False, loc='upper right')
ax[1].set_title('(b) Energy to 75% accuracy, 95% CI', loc='left')
plt.tight_layout(); plt.savefig(fd + '/fig_convergence_v3.pdf'); plt.close()
print('ok')
