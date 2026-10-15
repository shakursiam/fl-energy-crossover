#!/usr/bin/env python3
"""Statistics for Campaign v2. Usage: python3 -I stats_campaign_v2.py <out dir with recomputed_runs.csv>"""
import sys, json, numpy as np, pandas as pd
out = sys.argv[1]
R = pd.read_csv(out + '/recomputed_runs.csv')
rng = np.random.default_rng(12345)

def ols(E, y):
    E = np.asarray(E, float); y = np.asarray(y, float)
    A = np.c_[E, np.ones_like(E)]
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (a*E + c); r2 = 1 - (res**2).sum()/((y-y.mean())**2).sum()
    return a, c, r2

def boot_fit(df, ycol, B=5000):
    # stratified bootstrap: resample runs within each E
    groups = [g[ycol].values for _, g in df.groupby('E')]
    Es = [g.E.iloc[0] for _, g in df.groupby('E')]
    out = []
    for _ in range(B):
        E = []; y = []
        for e, g in zip(Es, groups):
            s = rng.choice(g, len(g)); E += [e]*len(s); y += list(s)
        out.append(ols(E, y))
    return np.array(out)

def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

res = {}
# ---- A. reported numbers reproduced from run_summary (all 60 main runs)
M = R[R.step == 'step1_main']
a, c, r2 = ols(M.E, M.s_Etr)
res['summary_main60'] = dict(slope=a, intercept=c, r2=r2, mean_comm_summary=float(M.s_Ecomm.mean()),
                             estar_A=float((M.s_Ecomm.mean()-c)/a), n=len(M))
# ---- B. recomputed on the traces that are present
Mp = M[M.trace_present]
a, c, r2 = ols(Mp.E, Mp.Etr)
bf = boot_fit(Mp, 'Etr')
res['recomputed_main_present'] = dict(n=len(Mp), slope=a, slope_ci=ci(bf[:, 0]), intercept=c, intercept_ci=ci(bf[:, 1]), r2=r2,
    per_E_n=Mp.groupby('E').size().to_dict())
Mq = Mp[Mp.nseg == 3]
res['recomputed_main_present']['mean_Ecomm_recomputed'] = float(Mq.Ecomm.mean())
res['recomputed_main_present']['mean_Ecomm_summary_same_runs'] = float(Mq.s_Ecomm.mean())
res['recomputed_main_present']['mean_Ecomm_all_phases_recomputed'] = float(Mq.Ecomm_all.mean())
res['recomputed_main_present']['median_Ecomm_recomputed'] = float(Mq.Ecomm.median())
res['recomputed_main_present']['mean_Tcomm'] = float(Mq.Tcomm.mean())
res['recomputed_main_present']['n_stall_gt500ms'] = int((Mq.Tul > 500).sum())
# ---- C. Does the comm energy summary differ from the stated baseline? (all present traces)
Z = R[(R.trace_present) & (R.nseg == 3)]
res['comm_summary_vs_recomputed'] = dict(n=len(Z), mean_diff_mJ=float((Z.s_Ecomm-Z.Ecomm).mean()),
    mean_diff_mW_per_window=float(((Z.s_Ecomm-Z.Ecomm)/Z.Tcomm*1000).mean()),
    max_diff_mJ=float((Z.s_Ecomm-Z.Ecomm).max()),
    comp_max_abs_diff_mJ=float((R[R.trace_present].Etr-R[R.trace_present].s_Etr).abs().max()),
    n_comp_exact=int(((R[R.trace_present].Etr-R[R.trace_present].s_Etr).abs() < 0.002).sum()))
# ---- D. per-E means (recomputed, present only)
res['per_E_main_present'] = {int(e): dict(n=int(len(g)), Etr_mean=float(g.Etr.mean()), Etr_sd=float(g.Etr.std(ddof=1)) if len(g) > 1 else None,
    Ecomm_mean=float(g[g.nseg == 3].Ecomm.mean()), Ecomm_sd=float(g[g.nseg == 3].Ecomm.std(ddof=1)), Ecomm_median=float(g[g.nseg == 3].Ecomm.median()),
    base_mean=float(g.base.mean())) for e, g in Mp.groupby('E')}
# ---- E. Step 2 tail A/B, recomputed and summary
T = R[R.step == 'step2_tail']
tail = {}
for dly, g in T.groupby('delay'):
    a, c, r2 = ols(g.E, g.Etr); bf = boot_fit(g, 'Etr')
    a_s, c_s, r2_s = ols(g.E, g.s_Etr)
    gq = g[g.nseg == 3]
    tail[int(dly)] = dict(n=len(g), slope=a, slope_ci=ci(bf[:, 0]), intercept=c, intercept_ci=ci(bf[:, 1]), r2=r2,
        summary_slope=a_s, summary_intercept=c_s, mean_Ecomm=float(gq.Ecomm.mean()), mean_Ecomm_summary=float(gq.s_Ecomm.mean()),
        mean_Ecomm_all=float(gq.Ecomm_all.mean()), n_comm=len(gq))
res['tail_ab'] = tail
# paired difference of intercepts with stratified bootstrap and permutation test
def c0(g): return ols(g.E, g.Etr)[1]
g0 = T[T.delay == 0]; g60 = T[T.delay == 60]; g200 = T[T.delay == 200]
def boot_diff(ga, gb, B=5000):
    d = []
    for _ in range(B):
        def s(g): 
            E = []; y = []
            for e, h in g.groupby('E'):
                v = rng.choice(h.Etr.values, len(h)); E += [e]*len(v); y += list(v)
            return ols(E, y)[1]
        d.append(s(ga)-s(gb))
    return np.array(d)
for nm, ga, gb in [('0_minus_60', g0, g60), ('200_minus_60', g200, g60), ('0_minus_200', g0, g200)]:
    d = boot_diff(ga, gb)
    # permutation: shuffle delay labels within each E
    comb = pd.concat([ga.assign(lab=1), gb.assign(lab=0)])
    obs = c0(ga)-c0(gb); cnt = 0; P = 5000
    for _ in range(P):
        parts = []
        for e, h in comb.groupby('E'):
            lab = rng.permutation(h.lab.values); parts.append(h.assign(lab=lab))
        h = pd.concat(parts)
        dd = c0(h[h.lab == 1]) - c0(h[h.lab == 0])
        if abs(dd) >= abs(obs): cnt += 1
    res.setdefault('tail_intercept_diff', {})[nm] = dict(diff=float(obs), ci=ci(d), perm_p=float((cnt+1)/(P+1)))
# per-cell train-energy means and the leak (excess over 60 ms cell) at each E
cell = T.groupby(['delay', 'E']).Etr.agg(['mean', 'std', 'count']).reset_index()
res['tail_cell_means'] = cell.to_dict('records')
leak = {}
for e in [1, 5, 20]:
    m0 = T[(T.delay == 0) & (T.E == e)].Etr; m60 = T[(T.delay == 60) & (T.E == e)].Etr; m200 = T[(T.delay == 200) & (T.E == e)].Etr
    leak[e] = dict(d0_minus_d60=float(m0.mean()-m60.mean()), d200_minus_d60=float(m200.mean()-m60.mean()),
                   welch_t_0v60=float((m0.mean()-m60.mean())/np.sqrt(m0.var(ddof=1)/len(m0)+m60.var(ddof=1)/len(m60))))
res['tail_leak_by_E'] = leak
# ---- F. Same 60 ms setting measured twice: step 1 (present) vs step 2 (delay 60), comm energy
sc = []
for e in [1, 5, 20]:
    a1 = M[(M.E == e) & M.trace_present & (M.nseg == 3)].Ecomm; a2 = T[(T.delay == 60) & (T.E == e) & (T.nseg == 3)].Ecomm
    sc.append(dict(E=e, n_step1=len(a1), mean_step1=float(a1.mean()), median_step1=float(a1.median()),
                   n_step2=len(a2), mean_step2=float(a2.mean()), median_step2=float(a2.median())))
res['same_setting_step1_vs_step2_comm'] = sc
s1 = M[M.E.isin([1, 5, 20])].s_Ecomm; s2 = T[(T.delay == 60)].s_Ecomm
res['same_setting_summary_comm'] = dict(step1_mean=float(s1.mean()), step2_mean=float(s2.mean()), n1=len(s1), n2=len(s2),
    welch_t=float((s1.mean()-s2.mean())/np.sqrt(s1.var(ddof=1)/len(s1)+s2.var(ddof=1)/len(s2))))
# ---- G. comm energy vs RF: heavy tail
allp = R[(R.trace_present) & (R.nseg == 3)]
res['comm_distribution_recomputed'] = dict(n=len(allp), mean=float(allp.Ecomm.mean()), median=float(allp.Ecomm.median()),
    p10=float(allp.Ecomm.quantile(.1)), p90=float(allp.Ecomm.quantile(.9)), mean_all_phases=float(allp.Ecomm_all.mean()),
    mean_cfg=float(allp.Ecfg.mean()), n_Tul_gt_500=int((allp.Tul > 500).sum()), n_Tul_gt_1000=int((allp.Tul > 1000).sum()),
    Tcomm_median=float(allp.Tcomm.median()), Tcomm_mean=float(allp.Tcomm.mean()),
    corr_Ecomm_Tcomm=float(np.corrcoef(allp.Ecomm, allp.Tcomm)[0, 1]))
# ---- H. baseline
pr = R[R.trace_present]
res['baseline'] = dict(n=len(pr), median_of_run_baselines=float(pr.base.median()), mean=float(pr.base.mean()), min=float(pr.base.min()), max=float(pr.base.max()),
    drift_mean=float(pr.drift.mean()), drift_max=float(pr.drift.max()), mean_vs_median_offset_mW=float(((pr.mean_base_pre+pr.mean_base_post)/2-pr.base).mean()),
    by_step={k: float(v) for k, v in pr.groupby('step').base.mean().items()})
# ---- I. gap (post-communication) energies at 60 ms and 200 ms
gp = R[(R.trace_present) & (R.nseg == 3) & (R.delay > 0)]
res['gap_energy'] = {int(dl): dict(n=int(len(g)), after_cfg=float(g.gap_after_cfg_E.mean()), after_dl=float(g.gap_after_dl_E.mean()),
    after_train=float(g.gap_after_train_E.mean()), after_cfg_sd=float(g.gap_after_cfg_E.std(ddof=1)),
    after_dl_sd=float(g.gap_after_dl_E.std(ddof=1)), after_train_sd=float(g.gap_after_train_E.std(ddof=1))) for dl, g in gp.groupby('delay')}
json.dump(res, open(out + '/stats.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
