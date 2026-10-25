#!/usr/bin/env python3
"""Analytical baselines of Section IV. Usage: python3 -I analytical_baselines.py <data_summary dir>
Reads paper_numbers.json (measured epoch time and plateau power) and writes analytical_baselines.json.
Each model gives a nominal value and a range from the stated parameter bounds."""
import sys, json, numpy as np
ds = sys.argv[1]; N = json.load(open(ds + '/paper_numbers.json'))
t_ep = N['t_ep']['slope_ms'] / 1000.0          # measured epoch time (s)
bits = 2 * 3120 * 8                             # download + upload of the 780-parameter head
f = 240e6                                       # CPU clock (Hz)
C = t_ep * f                                    # cycles per epoch
rate = lambda B, snr_db: B * np.log2(1 + 10 ** (snr_db / 10))
# Shannon capacity: P_tx * bits / R (mW * s = mJ); nominal 20 MHz, 20 dB, 150 mW
sh = dict(nom=150 * bits / rate(20e6, 20), lo=100 * bits / rate(40e6, 30), hi=200 * bits / rate(20e6, 10))
# Switching energy: kappa f^2 C (J -> mJ); kappa 5e-28 (1e-28 to 1e-27)
sw = {k: kap * f ** 2 * C * 1000 for k, kap in (('nom', 5e-28), ('lo', 1e-28), ('hi', 1e-27))}
comb = dict(nom=sh['nom'] / sw['nom'], lo=sh['lo'] / sw['hi'], hi=sh['hi'] / sw['lo'])
# Datasheet currents: effective radio power 536 mW (450-650) for 20 ms (15-25); CPU power 132 mW (100-160) per epoch
dsc = dict(nom=536 * 0.020, lo=450 * 0.015, hi=650 * 0.025)
dsp = dict(nom=132 * t_ep, lo=100 * t_ep, hi=160 * t_ep)
dse = dict(nom=dsc['nom'] / dsp['nom'], lo=dsc['lo'] / dsp['hi'], hi=dsc['hi'] / dsp['lo'])
out = dict(t_ep_ms=t_ep * 1000, cycles=C, shannon=sh, switching=sw, combined=comb, ds_comm=dsc, ds_comp=dsp, ds_Estar=dse)
json.dump(out, open(ds + '/analytical_baselines.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
