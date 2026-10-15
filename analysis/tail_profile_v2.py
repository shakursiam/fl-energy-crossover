#!/usr/bin/env python3
"""Tail profiles from raw_v2. Usage: python3 -I tail_profile_v2.py <raw_v2 dir> <out dir>"""
import sys, os, json, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
R = pd.read_csv(os.path.join(out, 'recomputed_runs.csv'))
R = R[R.trace_present]
def segs_nz(m):
    idx = np.where(m)[0]; s = []; a = pv = idx[0]
    for i in idx[1:]:
        if i-pv > 1: s.append((a, pv)); a = i
        pv = i
    s.append((a, pv)); return s
prof = {'after_cfg': [], 'after_dl': [], 'after_train': [], 'after_ul': []}
trainstart = {0: [], 60: [], 200: []}
for _, r in R.iterrows():
    sub = 'step1_main' if r.step == 'step1_main' else 'step2_tail'
    d = pd.read_csv(f'{root}/{sub}/{r.run_id}/power_trace.csv'); p = d.power_mW.values; rw = d.raw_phase_code.values
    base = r.base
    tr = np.where(rw == 6)[0]; t0, t1 = tr[0], tr[-1]
    # first 40 ms of the training window (aligned to train start), excess over baseline
    trainstart[int(r.delay)].append((p[t0:t0+40]-base).tolist() if r.Ttr >= 40 else None)
    if r.nseg != 3 or r.delay != 60: continue
    S = segs_nz((rw >= 1) & (rw <= 5))
    (c0, c1), (d0, d1), (u0, u1) = S
    prof['after_cfg'].append((p[c1+1:c1+61]-base).tolist())
    prof['after_dl'].append((p[d1+1:d1+61]-base).tolist())
    prof['after_train'].append((p[t1+1:t1+61]-base).tolist())
    n_after = len(p)-(u1+1)
    if n_after >= 200: prof['after_ul'].append((p[u1+1:u1+201]-base).tolist())
res = {}
for k, v in prof.items():
    if not v: continue
    a = np.array(v); m = a.mean(0)
    res[k] = dict(n=len(v), mean_excess_mW_ms=[float(x) for x in m],
                  energy_first_20ms_mJ=float(m[:20].sum()/1000), energy_first_60ms_mJ=float(m[:60].sum()/1000),
                  energy_total_mJ=float(m.sum()/1000), per_run_energy_60ms_mJ=[float(x) for x in a[:, :60].sum(1)/1000])
# training-window start excess: mean excess power in ms 0-9, 10-24, 25-39 for each delay, relative to the steady training excess (ms 40+ not available) -> use E>=5 only
ts = {}
for dl, lst in trainstart.items():
    L = [np.array(x) for x in lst if x is not None]
    if not L: continue
    a = np.array(L); ts[dl] = dict(n=len(L), mean_excess=[float(x) for x in a.mean(0)])
res['trainstart'] = ts
json.dump(res, open(os.path.join(out, 'tail_profile.json'), 'w'))
for k in ['after_cfg', 'after_dl', 'after_train', 'after_ul']:
    if k in res: print(k, res[k]['n'], 'E20ms %.3f E60ms %.3f Etotal %.3f mJ' % (res[k]['energy_first_20ms_mJ'], res[k]['energy_first_60ms_mJ'], res[k]['energy_total_mJ']),
        'first5ms excess mW', np.round(res[k]['mean_excess_mW_ms'][:5], 0), ' t=20-40ms mean', round(float(np.mean(res[k]['mean_excess_mW_ms'][20:40])), 1))
for dl, v in ts.items():
    m = np.array(v['mean_excess']); print('train start delay', dl, 'n', v['n'], 'ms0-9 mean %.1f, ms10-24 %.1f, ms25-39 %.1f' % (m[:10].mean(), m[10:25].mean(), m[25:40].mean()))
