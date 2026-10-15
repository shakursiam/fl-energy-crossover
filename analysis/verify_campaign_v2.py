#!/usr/bin/env python3
"""Independent re-analysis of Campaign v2 raw traces.
Usage: python3 -I verify_campaign_v2.py <path to data/raw_v2> <output dir>
Baseline: median of idle samples [5 s, 25 s) before the first marker and
[5 s, 25 s) after the last marker, averaged (this reproduces run_summary.csv).
Energy above baseline = sum((P - base) * 1 ms). Phases come from raw_phase_code.
"""
import os, sys, json
import numpy as np, pandas as pd

root, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
S = pd.read_csv(os.path.join(root, 'run_summary.csv'))

def nz_runs(mask):
    idx = np.where(mask)[0]
    segs = []; s0 = pv = idx[0]
    for i in idx[1:]:
        if i - pv > 1:
            segs.append((s0, pv)); s0 = i
        pv = i
    segs.append((s0, pv)); return segs

rows = []
for _, r in S.iterrows():
    sub = 'step1_main' if r.campaign_step == 'step1_main' else 'step2_tail'
    f = os.path.join(root, sub, r.run_id, 'power_trace.csv')
    rec = dict(run_id=r.run_id, step=r.campaign_step, E=int(r.E), delay=int(r.delay_ms), rep=int(r.rep),
               trace_present=os.path.exists(f),
               s_Etr=r.marginal_comp_energy_mJ, s_Ecomm=r.marginal_comm_energy_mJ,
               s_Ecfg=r.energy_config_mJ, s_Edl=r.energy_download_mJ, s_Eul=r.energy_upload_mJ,
               s_base=r.baseline_mean_mW, s_gaps=r.energy_gaps_mJ, s_Tcomm=r.T_comm_ms)
    if rec['trace_present']:
        d = pd.read_csv(f); p = d.power_mW.values; rw = d.raw_phase_code.values; t = d.timestamp_ms.values
        rec['mono'] = bool((np.diff(t) > 0).all()); rec['dropped'] = int(d.dropped_samples.sum())
        act = np.where(rw != 0)[0]; a, b = act[0], act[-1]
        bp = np.median(p[5000:25000]); bq = np.median(p[b+5000:b+25000]); base = (bp+bq)/2
        rec.update(base=base, base_pre=bp, base_post=bq, drift=abs(bp-bq)/base*100,
                   mean_base_pre=p[5000:25000].mean(), mean_base_post=p[b+5000:b+25000].mean())
        en = lambda x, y: float((p[x:y+1]-base).sum()/1000.0)
        rec['Etr'] = float((p[rw == 6]-base).sum()/1000.0); rec['Ttr'] = int((rw == 6).sum())
        tr = np.where(rw == 6)[0]; tr0, tr1 = tr[0], tr[-1]
        comm = nz_runs((rw >= 1) & (rw <= 5))
        rec['nseg'] = len(comm)
        if len(comm) == 3:
            (c0, c1), (d0, d1), (u0, u1) = comm
            rec.update(Ecfg=en(c0, c1), Edl=en(d0, d1), Eul=en(u0, u1),
                       Tcfg=c1-c0+1, Tdl=d1-d0+1, Tul=u1-u0+1)
            rec['Ecomm'] = rec['Edl'] + rec['Eul']; rec['Ecomm_all'] = rec['Ecomm'] + rec['Ecfg']
            rec['Tcomm'] = rec['Tdl'] + rec['Tul']
            # post-communication windows (idle samples directly after each comm segment)
            for nm, e1 in [('cfg', c1), ('dl', d1)]:
                nxt = {'cfg': d0, 'dl': tr0}[nm]
                gap = p[e1+1:nxt]
                rec[f'gap_after_{nm}_n'] = len(gap)
                rec[f'gap_after_{nm}_E'] = float((gap-base).sum()/1000.0) if len(gap) else np.nan
            gap = p[tr1+1:u0]; rec['gap_after_train_n'] = len(gap)
            rec['gap_after_train_E'] = float((gap-base).sum()/1000.0) if len(gap) else np.nan
            # tail after upload (first 200 ms of idle after the last marker)
            rec['post_ul_200ms_E'] = float((p[u1+1:u1+201]-base).sum()/1000.0)
            # profile: mean excess power in 10 ms bins after each comm end, only for gap >= 60 ms
            if rec['gap_after_dl_n'] >= 60: rec['_prof_dl'] = (p[d1+1:d1+61]-base).tolist()
            if rec['gap_after_train_n'] >= 60: rec['_prof_tr'] = (p[tr1+1:tr1+61]-base).tolist()
            if rec['gap_after_cfg_n'] >= 60: rec['_prof_cfg'] = (p[c1+1:c1+61]-base).tolist()
    rows.append(rec)
R = pd.DataFrame(rows)
prof = {k: R[k].dropna().tolist() for k in ['_prof_dl', '_prof_tr', '_prof_cfg'] if k in R}
R = R.drop(columns=[c for c in R.columns if c.startswith('_prof')])
R.to_csv(os.path.join(outdir, 'recomputed_runs.csv'), index=False)
json.dump(prof, open(os.path.join(outdir, 'gap_profiles.json'), 'w'))
print('traces present:', int(R.trace_present.sum()), 'of', len(R))
