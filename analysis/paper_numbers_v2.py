#!/usr/bin/env python3
"""All numbers used in the manuscript, from run_summary.csv (compute columns are verified against raw traces).
Usage: python3 -I paper_numbers_v2.py <raw_v2 dir> <out dir>"""
import sys, json, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
S = pd.read_csv(root + '/run_summary.csv')
rng = np.random.default_rng(777)
def ols(E, y):
    E = np.asarray(E, float); y = np.asarray(y, float); A = np.c_[E, np.ones_like(E)]
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None); r = y-(a*E+c); return a, c, 1-(r**2).sum()/((y-y.mean())**2).sum()
M = S[S.campaign_step == 'step1_main']; T60 = S[(S.campaign_step == 'step2_tail') & (S.delay_ms == 60)]
N = {}
# per-E table (step 1)
rows = []
for e, g in M.groupby('E'):
    rows.append(dict(E=int(e), n=len(g), Ttr=g.dur_train_ms.mean(), Ttr_sd=g.dur_train_ms.std(), comp=g.marginal_comp_energy_mJ.mean(), comp_sd=g.marginal_comp_energy_mJ.std(),
        comm=g.marginal_comm_energy_mJ.mean(), comm_sd=g.marginal_comm_energy_mJ.std(), comm_med=g.marginal_comm_energy_mJ.median(),
        Tcomm_med=g.T_comm_ms.median(), Tcomm_mean=g.T_comm_ms.mean(), cfg=g.energy_config_mJ.mean(), base=g.baseline_mean_mW.mean()))
N['perE'] = rows
a, c, r2 = ols(M.E, M.marginal_comp_energy_mJ); N['fit'] = dict(a=a, c=c, r2=r2)
comm = M.marginal_comm_energy_mJ.mean(); cfg = M.energy_config_mJ.mean()
N['comm_mean'] = comm; N['comm_median'] = M.marginal_comm_energy_mJ.median(); N['cfg_mean'] = cfg
N['Estar_A'] = (comm-c)/a; N['Estar_B'] = (comm+0.506-c)/a; N['Estar_all_cfg'] = (comm+cfg-c)/a
N['Estar_median_comm'] = (N['comm_median']-c)/a
# bootstrap E* (stratified over E for comp fit; comm resampled over all runs)
B = 5000; ea = []; eb = []; ec = []
groups = {e: g for e, g in M.groupby('E')}
for _ in range(B):
    Es = []; ys = []; cm = []; cf = []
    for e, g in groups.items():
        idx = rng.integers(0, len(g), len(g)); s = g.iloc[idx]
        Es += [e]*len(s); ys += list(s.marginal_comp_energy_mJ); cm += list(s.marginal_comm_energy_mJ); cf += list(s.energy_config_mJ)
    aa, cc, _ = ols(Es, ys); ea.append((np.mean(cm)-cc)/aa); eb.append((np.mean(cm)+0.506-cc)/aa); ec.append((np.mean(cm)+np.mean(cf)-cc)/aa)
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
N['Estar_A_ci'] = ci(ea); N['Estar_B_ci'] = ci(eb); N['Estar_all_cfg_ci'] = ci(ec)
# step-2 60 ms rows
a2, c2, r22 = ols(T60.E, T60.marginal_comp_energy_mJ); comm2 = T60.marginal_comm_energy_mJ.mean()
ea2 = []
g2 = {e: g for e, g in T60.groupby('E')}
for _ in range(B):
    Es = []; ys = []; cm = []
    for e, g in g2.items():
        s = g.iloc[rng.integers(0, len(g), len(g))]; Es += [e]*len(s); ys += list(s.marginal_comp_energy_mJ); cm += list(s.marginal_comm_energy_mJ)
    aa, cc, _ = ols(Es, ys); ea2.append((np.mean(cm)-cc)/aa)
N['step2_60'] = dict(a=a2, c=c2, comm=comm2, Estar=(comm2-c2)/a2, Estar_ci=ci(ea2), n=len(T60))
# pooled comm mean over both sessions at 60 ms
N['comm_pooled_75'] = float(pd.concat([M.marginal_comm_energy_mJ, T60.marginal_comm_energy_mJ]).mean())
# session difference: Welch t on E in {1,5,20}
s1 = M[M.E.isin([1, 5, 20])].marginal_comm_energy_mJ; s2 = T60.marginal_comm_energy_mJ
N['session_diff'] = dict(m1=float(s1.mean()), m2=float(s2.mean()), t=float((s1.mean()-s2.mean())/np.sqrt(s1.var()/len(s1)+s2.var()/len(s2))),
    n1=len(s1), n2=len(s2))
# rank test (no normality assumption)
from itertools import combinations
x = np.r_[s1.values, s2.values]; obs = s1.mean()-s2.mean(); cnt = 0
for _ in range(20000):
    p = rng.permutation(x); cnt += abs(p[:len(s1)].mean()-p[len(s1):].mean()) >= abs(obs)
N['session_diff']['perm_p'] = float((cnt+1)/20001)
# timing
Tc = S.T_comm_ms
N['Tcomm'] = dict(median=float(Tc.median()), q25=float(Tc.quantile(.25)), q75=float(Tc.quantile(.75)), mean=float(Tc.mean()), n_gt500=int((Tc > 500).sum()), n=len(Tc),
                  max=float(Tc.max()))
d60 = S[S.delay_ms > 0]; ta, tb, _ = ols(S[S.delay_ms == 60].E, S[S.delay_ms == 60].dur_train_ms)
N['t_ep'] = dict(slope_ms=float(ta), intercept_ms=float(tb))
N['baseline'] = dict(mean=float(S.baseline_mean_mW.mean()), sd=float(S.baseline_mean_mW.std()), min=float(S.baseline_mean_mW.min()), max=float(S.baseline_mean_mW.max()),
    drift_mean=float(S.drift_pct.mean()), drift_max=float(S.drift_pct.max()), step1_mean=float(M.baseline_mean_mW.mean()))
N['dl_ul'] = dict(dl=float(M.energy_download_mJ.mean()), ul=float(M.energy_upload_mJ.mean()), Tdl=float(M.dur_download_ms.mean()), Tul_med=float(M.dur_upload_ms.median()),
    Tdl_med=float(M.dur_download_ms.median()))
json.dump(N, open(out + '/paper_numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if k != 'perE'}, indent=1, default=float))
for r in rows: print({k: round(v, 3) for k, v in r.items()})
