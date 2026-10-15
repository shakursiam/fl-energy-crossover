#!/usr/bin/env python3
"""Write run_summary_unclamped.csv: communication energies re-integrated WITHOUT clamping negative samples.
Phase windows follow run_summary.csv (config, download, upload sample counts of raw phase codes 1..5, in time order).
Usage: python3 -I make_unclamped_summary.py <raw_v2 dir> <out csv>"""
import sys, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
S = pd.read_csv(root + '/run_summary.csv'); rows = []
for i, r in S.iterrows():
    sub = 'step1_main' if r.campaign_step == 'step1_main' else 'step2_tail'
    d = pd.read_csv(f'{root}/{sub}/{r.run_id}/power_trace.csv'); p = d.power_mW.values; rw = d.raw_phase_code.values
    b = np.where(rw != 0)[0][-1]
    base = (np.median(p[5000:25000]) + np.median(p[b+5000:b+25000])) / 2
    idx = np.where((rw >= 1) & (rw <= 5))[0]
    nc, nd, nu = int(r.dur_config_ms), int(r.dur_download_ms), int(r.dur_upload_ms)
    if nc + nd + nu != len(idx):   # fall back to counting from both ends
        pass
    wc, wd, wu = idx[:nc], idx[nc:nc+nd], idx[-nu:]
    e = lambda w, clamp: float((np.maximum(p[w]-base, 0) if clamp else p[w]-base).sum()/1000.0)
    rows.append(dict(i=i, cfg_u=e(wc, 0), dl_u=e(wd, 0), ul_u=e(wu, 0), cfg_c=e(wc, 1), dl_c=e(wd, 1), ul_c=e(wu, 1), n_idx=len(idx), n_sum=nc+nd+nu))
R = pd.DataFrame(rows).set_index('i')
chk = (R.dl_c + R.ul_c - S.marginal_comm_energy_mJ).abs()
print('clamped window reproduces summary comm energy: %d of %d within 0.002 mJ, max %.4f' % ((chk < 0.002).sum(), len(S), chk.max()))
print('cfg: %d of %d within 0.002' % (((R.cfg_c - S.energy_config_mJ).abs() < 0.002).sum(), len(S)))
print('sample-count mismatch runs:', int((R.n_idx != R.n_sum).sum()))
S2 = S.copy()
S2['energy_config_mJ'] = R.cfg_u; S2['energy_download_mJ'] = R.dl_u; S2['energy_upload_mJ'] = R.ul_u
S2['marginal_comm_energy_mJ'] = R.dl_u + R.ul_u
S2.to_csv(out, index=False)
print('mean comm clamped %.3f unclamped %.3f' % ((R.dl_c+R.ul_c).mean(), (R.dl_u+R.ul_u).mean()))
