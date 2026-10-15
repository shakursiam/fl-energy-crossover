#!/usr/bin/env python3
"""Derived quantities (sleep, contention, convergence energy, model mismatch). Usage: python3 -I derived_v2.py <out dir>"""
import sys, json, numpy as np
out = sys.argv[1]; N = json.load(open(out + '/paper_numbers.json'))
a = N['fit']['a']; c0 = N['fit']['c']; dEc = N['comm_mean']; Pb = N['baseline']['mean']; tep = N['t_ep']['slope_ms']; Tcm = N['Tcomm']['median']
D = dict(a=a, c0=c0, dEc=dEc, Pb=Pb, tep=tep, Tcomm_med=Tcm)
states = [('Standby', Pb, 0.0), ('Modem sleep', 100.0, 0.0), ('Light sleep', 10.0, 0.5), ('Deep sleep', 0.08, 180.0)]
def Estar(Ps, Tc_ms, dE=dEc): return ((Pb-Ps)*Tc_ms/1000 + dE - c0)/((Pb-Ps)*tep/1000 + a)
def Eturn(E, Ps, wake, Tc_ms, Tr=60.0):
    Tact = (Tc_ms + tep*E)/1000
    return (Pb*Tact + (a*E + c0 + dEc) + Ps*(Tr-Tact) + wake*1.0)/1000 if False else (Pb*Tact*1.0 + (a*E+c0+dEc)/1000*1000/1000*0)  # placeholder, replaced below
# explicit in mJ: mW*s = mJ
def Eturn_mJ(E, Ps, wake, Tc_ms, Tr=60.0):
    Tact = (Tc_ms + tep*E)/1000.0
    return Pb*Tact + (a*E + c0 + dEc) + Ps*(Tr-Tact) + wake
D['sleep'] = []
for nm, Ps, wk in states:
    D['sleep'].append(dict(state=nm, Ps=Ps, wake=wk, Estar_85=Estar(Ps, 85), Estar_125=Estar(Ps, Tcm), Estar_160=Estar(Ps, 160),
        Eturn_J={E: Eturn_mJ(E, Ps, wk, Tcm)/1000 for E in (1, 5, 20)}))
D['contention'] = {k: (k*dEc - c0)/a for k in (1.0, 1.5, 2.0, 3.0)}
R = {1: 5.42, 2: 3.75, 5: 2.05, 8: 1.82, 10: 1.80, 20: 1.40}
D['conv'] = []
for E, r in R.items():
    marg = r*(a*E + c0 + dEc)/1000
    Tact = (Tcm + tep*E)/1000
    gross = r*(Pb*Tact + a*E + c0 + dEc)/1000
    deep = r*Eturn_mJ(E, 0.08, 180.0, Tcm)/1000
    D['conv'].append(dict(E=E, R=r, traffic_kB=r*6240/1000, accesses=r*2, marg=marg, gross=gross, deep=deep))
D['conv_best'] = dict(marg=min(D['conv'], key=lambda x: x['marg'])['E'], gross=min(D['conv'], key=lambda x: x['gross'])['E'], deep=min(D['conv'], key=lambda x: x['deep'])['E'])
# mismatch relative to v2 reference
ref = dict(comm=dEc, comp=a, Estar=N['Estar_A'])
pred = dict(shannon_comm=0.056, switch_comp=0.053, comb_Estar=1.07, ds_comm=10.73, ds_comp=1.005, ds_Estar=10.67, cd_Estar=500.0)
err = lambda p, r: (p-r)/r*100
D['mismatch'] = dict(ref=ref, shannon=err(0.056, ref['comm']), switching=err(0.053, ref['comp']), combined=err(1.07, ref['Estar']),
    ds_comm=err(10.73, ref['comm']), ds_comp=err(1.012, ref['comp']), ds_Estar=err(10.6, ref['Estar']), comm_dom=err(500.0, ref['Estar']), comm_dom_ratio=500/ref['Estar'])
D['awake_vs_marginal_E1'] = dict(awake_mJ=Pb*(Tcm + tep)/1000, marginal_mJ=a+c0+dEc)
D['epoch_gross_mJ'] = Pb*tep/1000
json.dump(D, open(out + '/derived.json', 'w'), indent=1, default=float)
print(json.dumps(D, indent=1, default=float))
