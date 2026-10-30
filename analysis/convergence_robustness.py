#!/usr/bin/env python3
"""Robustness of the energy-optimal E (Section VIII). Usage: python3 -I convergence_robustness.py <data_summary dir>
Reads the saved test-accuracy curves (convergence_curves.npz) and derived.json; no retraining.
(1) Rounds to 75-78% test accuracy and the energy-optimal E for each accounting.
(2) Seeds that reach 75% within the first round (R < 1).
(3) Deep-sleep optimum for the measured wake-up energy, its 95% CI bounds, and 2x the measurement.
Writes <data_summary>/convergence_robustness.json."""
import sys, json, numpy as np
ds = sys.argv[1]
z = np.load(ds + '/convergence_curves.npz'); D = json.load(open(ds + '/derived.json'))
a, c0, dEc, Pb, tep, Tc = D['a'], D['c0'], D['dEc'], D['Pb'], D['tep'], D['Tcomm_med']
EPOCHS = [1, 2, 5, 8, 10, 20]
TR_S = 60.0
_deep = [s for s in D['sleep'] if s['state'] == 'Deep sleep'][0]
PS_DEEP_MW, EWAKE_MJ = _deep['Ps'], _deep['wake']   # Campaign 4 (measured) or assumed, see derived.json

def rounds_to(h, target):
    """Same interpolation as convergence_v3.py."""
    for r in range(1, len(h)):
        if h[r] >= target:
            return r - 1 + (target - h[r - 1]) / (h[r] - h[r - 1]) if h[r] > h[r - 1] else float(r)
    return np.nan

def turn_mj(E, ewake=EWAKE_MJ):
    marg = dEc + a * E + c0; tact = Tc + tep * E
    gross = Pb * tact / 1000 + marg
    return marg, gross, gross + PS_DEEP_MW * (TR_S * 1000 - tact) / 1000 + ewake

out = {'targets': {}, 'wake': {}}
for t in [75.0, 76.0, 77.0, 78.0]:
    R = {e: np.array([rounds_to(h, t) for h in z[f'E{e}']]) for e in EPOCHS}
    Rm = {e: float(np.nanmean(R[e])) for e in EPOCHS}
    out['targets'][str(t)] = dict(R_mean={str(e): Rm[e] for e in EPOCHS},
                                  reached={str(e): int((~np.isnan(R[e])).sum()) for e in EPOCHS},
                                  below_one_round={str(e): int((R[e] < 1).sum()) for e in EPOCHS},
                                  opt={k: min(EPOCHS, key=lambda e: Rm[e] * turn_mj(e)[i]) for i, k in enumerate(['marg', 'gross', 'deep'])})
R75 = out['targets']['75.0']['R_mean']
try:
    _ci = json.load(open(ds + '/sleep_campaign.json'))['summary']['E_wake_deep_mJ']['ci95']
except FileNotFoundError:
    _ci = [EWAKE_MJ, EWAKE_MJ]
for ew in [EWAKE_MJ, _ci[0], _ci[1], 2 * EWAKE_MJ]:
    J = {e: R75[str(e)] * turn_mj(e, ew)[2] for e in EPOCHS}
    out['wake'][f'{ew:.1f}'] = dict(J_mJ={str(e): J[e] for e in EPOCHS}, opt=min(J, key=J.get))
json.dump(out, open(ds + '/convergence_robustness.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
