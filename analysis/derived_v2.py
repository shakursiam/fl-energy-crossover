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
    D['sleep'].append(dict(state=nm, Ps=Ps, wake=wk, Estar_q25=Estar(Ps, N['Tcomm']['q25']), Estar_125=Estar(Ps, Tcm), Estar_q75=Estar(Ps, N['Tcomm']['q75']),
        Eturn_J={E: Eturn_mJ(E, Ps, wk, Tcm)/1000 for E in (1, 5, 20)}))
D['contention'] = {k: (k*dEc - c0)/a for k in (1.0, 1.5, 2.0, 3.0)}
# rounds, convergence energies and the per-seed optimum are in convergence.json (convergence_v3.py)
# mismatch relative to the main campaign (model values from analytical_baselines.json)
AB = json.load(open(out + '/analytical_baselines.json'))
ref = dict(comm=dEc, comp=a, Estar=N['Estar_A'])
err = lambda p, r: (p-r)/r*100
D['mismatch'] = dict(ref=ref, shannon=err(AB['shannon']['nom'], ref['comm']), switching=err(AB['switching']['nom'], ref['comp']),
    combined=err(AB['combined']['nom'], ref['Estar']), ds_comm=err(AB['ds_comm']['nom'], ref['comm']),
    ds_comp=err(AB['ds_comp']['nom'], ref['comp']), ds_Estar=err(AB['ds_Estar']['nom'], ref['Estar']))
D['awake_vs_marginal_E1'] = dict(awake_mJ=Pb*(Tcm + tep)/1000, marginal_mJ=a+c0+dEc)
D['epoch_gross_mJ'] = Pb*tep/1000
json.dump(D, open(out + '/derived.json', 'w'), indent=1, default=float)
print(json.dumps(D, indent=1, default=float))
