"""Campaign 4 analysis: board-level sleep power and wake-up energy of the ESP32-S3 DUT.

Segments a power_trace.csv recorded by runner/run_sleep_campaign.py using the phase codes
emitted by firmware/BOARD_A_SLEEP/BOARD_A_SLEEP.ino and reports, per cycle and pooled:
  P_standby, P_modem, P_light, P_deep          plateau means (mW, edges trimmed)
  E_wake_deep,  T_wake_deep                     power rise after deep sleep -> IP obtained (gross)
  E_wake_light, T_wake_light                    light-sleep wake-up -> Wi-Fi re-associated (gross)
  E_entry_light, E_entry_deep                   delimiter pulse -> sleep floor reached (gross)

Usage: python analysis/analyze_sleep_campaign.py data/raw_sleep/session_01 [--out data_summary/sleep_campaign.json]
Reads power_trace.csv or power_trace.csv.gz in each session directory.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

trapezoid = getattr(np, 'trapezoid', None) or np.trapz   # numpy < 2.0 compatibility

TRIM_AWAKE_MS = 1000   # trimmed from both ends of the standby / modem windows
TRIM_SLEEP_MS = 500    # trimmed from both ends of the light / deep sleep windows
RISE_MW = 50.0         # wake-up / entry detection threshold above the sleep floor
PULSE_MIN, PULSE_MAX = 10, 40   # delimiter pulses are 20 samples long
HOLD_MIN = 100                  # held wake markers last ~1 s
SESSION_MIN = 400               # session markers last 500 samples


def runs(code):
    edges = np.flatnonzero(np.diff(code)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(code)] - 1
    return [(int(code[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def energy_mJ(t_ms, p_mW, a, b):
    """Gross energy over samples a..b inclusive (trapezoid on the logger timestamps)."""
    return float(trapezoid(p_mW[a:b + 1], t_ms[a:b + 1] / 1000.0))


def plateau(t, p, i, v, a, b, trim_ms):
    m = (t >= t[a] + trim_ms) & (t <= t[b] - trim_ms)
    idx = np.flatnonzero(m)
    idx = idx[(idx >= a) & (idx <= b)]
    return {'P_mW': float(p[idx].mean()), 'P_sd_mW': float(p[idx].std(ddof=1)),
            'I_mA': float(i[idx].mean() * 1000), 'V_V': float(v[idx].mean()),
            'n': int(len(idx)), 'window_ms': float(t[b] - t[a])}


def first_sustained(mask, start, k=5):
    """First index >= start where mask holds for k consecutive samples."""
    run = 0
    for j in range(start, len(mask)):
        run = run + 1 if mask[j] else 0
        if run == k:
            return j - k + 1
    return None


def segment(df):
    t = df.timestamp_ms.to_numpy(float)
    p = df.power_mW.to_numpy(float)
    i = df.current_A.to_numpy(float)
    v = df.voltage_V.to_numpy(float)
    code = df.raw_phase_code.to_numpy(int)
    R = runs(code)

    sess = [r for r in R if r[0] == 7 and r[2] - r[1] + 1 >= SESSION_MIN]
    if len(sess) < 2:
        raise SystemExit('session markers not found (need start and end 500 ms code-7 pulses)')
    s0, s1 = sess[0][2], sess[-1][1]
    R = [r for r in R if s0 < r[1] < s1]
    pulses = {c: [r for r in R if r[0] == c and PULSE_MIN <= r[2] - r[1] + 1 <= PULSE_MAX] for c in (1, 2, 3, 4)}
    holds = {c: [r for r in R if r[0] == c and r[2] - r[1] + 1 >= HOLD_MIN] for c in (5, 6)}
    flags = [r for r in R if r[0] == 7]

    n = len(pulses[4])
    counts = {'standby_pulses': len(pulses[1]), 'modem_pulses': len(pulses[2]), 'light_pulses': len(pulses[3]),
              'deep_pulses': n, 'deep_wake_holds': len(holds[5]), 'light_wake_holds': len(holds[6]),
              'error_flags': len(flags)}
    cycles = []
    for k in range(n):
        cyc = {'cycle': k + 1}
        for name, c, trim in (('standby', 1, TRIM_AWAKE_MS), ('modem', 2, TRIM_AWAKE_MS), ('light', 3, TRIM_SLEEP_MS)):
            pa, pb = pulses[c][2 * k], pulses[c][2 * k + 1]
            cyc[name] = plateau(t, p, i, v, pa[2] + 1, pb[1] - 1, trim)
        # light-sleep entry: pulse start -> floor reached
        la, lb = pulses[3][2 * k], pulses[3][2 * k + 1]
        floor_l = cyc['light']['P_mW']
        j = first_sustained(p < floor_l + RISE_MW, la[2] + 1)
        cyc['E_entry_light_mJ'] = energy_mJ(t, p, la[1], j) if j else None
        # light-sleep wake-up: held code 6 (Wi-Fi start, association, DHCP)
        h6 = holds[6][k]
        cyc['E_wake_light_mJ'] = energy_mJ(t, p, h6[1], h6[2])
        cyc['T_wake_light_ms'] = float(t[h6[2]] - t[h6[1]])
        # deep sleep: entry pulse -> floor -> timer wake-up (power rise) -> held code 5 ends
        d = pulses[4][k]
        h5 = holds[5][k]
        if not d[2] < h5[1]:
            raise SystemExit(f'cycle {k + 1}: deep-sleep wake hold precedes entry pulse')
        fl = np.median(p[d[2] + 1000:h5[1] - 3000])
        rise = first_sustained(p > fl + RISE_MW, d[2] + 1000)
        cyc['deep'] = plateau(t, p, i, v, d[2] + 1, rise - 1, TRIM_SLEEP_MS)
        j = first_sustained(p < fl + RISE_MW, d[2] + 1)
        cyc['E_entry_deep_mJ'] = energy_mJ(t, p, d[1], j)
        cyc['E_wake_deep_mJ'] = energy_mJ(t, p, rise, h5[2])
        cyc['T_wake_deep_ms'] = float(t[h5[2]] - t[rise])
        cyc['T_boot_ms'] = float(t[h5[1]] - t[rise])
        cyc['T_assoc_dhcp_ms'] = float(t[h5[2]] - t[h5[1]])
        cyc['P_wake_deep_mean_mW'] = cyc['E_wake_deep_mJ'] / (cyc['T_wake_deep_ms'] / 1000.0)
        cyc['T_deep_ms'] = float(t[rise] - t[d[2]])
        lo, hi = min(pulses[1][2 * k][1], d[1]), h5[2]
        cyc['dropped_samples'] = int(df.dropped_samples.iloc[hi] - df.dropped_samples.iloc[lo])
        cyc['max_gap_ms'] = float(np.diff(t[lo:hi + 1]).max())
        cycles.append(cyc)
    return cycles, counts


def summarize(x, rng, B=10000):
    x = np.asarray([v for v in x if v is not None], float)
    boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            'median': float(np.median(x)), 'min': float(x.min()), 'max': float(x.max()), 'n': int(len(x)),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session_dirs', nargs='+')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    all_cycles, sessions = [], {}
    for d in a.session_dirs:
        path = os.path.join(d, 'power_trace.csv')
        df = pd.read_csv(path if os.path.exists(path) else path + '.gz')
        cycles, counts = segment(df)
        for c in cycles:
            c['session'] = os.path.basename(os.path.normpath(d))
        sessions[os.path.basename(os.path.normpath(d))] = {'counts': counts, 'n_cycles': len(cycles),
                                                           'samples': int(len(df))}
        all_cycles += cycles

    gates = {
        'no_dropped_samples': all(c['dropped_samples'] == 0 for c in all_cycles),
        'max_gap_le_2ms': all(c['max_gap_ms'] <= 2 for c in all_cycles),
        'no_error_flags': all(s['counts']['error_flags'] == 0 for s in sessions.values()),
        'pulse_counts_consistent': all(
            s['counts']['standby_pulses'] == s['counts']['modem_pulses'] == s['counts']['light_pulses']
            == 2 * s['counts']['deep_pulses'] and s['counts']['deep_pulses'] == s['counts']['deep_wake_holds']
            == s['counts']['light_wake_holds'] for s in sessions.values()),
    }
    rng = np.random.default_rng(0)
    metrics = {
        'P_standby_mW': [c['standby']['P_mW'] for c in all_cycles],
        'P_modem_mW': [c['modem']['P_mW'] for c in all_cycles],
        'P_light_mW': [c['light']['P_mW'] for c in all_cycles],
        'P_deep_mW': [c['deep']['P_mW'] for c in all_cycles],
        'I_light_mA': [c['light']['I_mA'] for c in all_cycles],
        'I_deep_mA': [c['deep']['I_mA'] for c in all_cycles],
        'V_bus_V': [c['standby']['V_V'] for c in all_cycles],
        'E_wake_deep_mJ': [c['E_wake_deep_mJ'] for c in all_cycles],
        'T_wake_deep_ms': [c['T_wake_deep_ms'] for c in all_cycles],
        'T_boot_ms': [c['T_boot_ms'] for c in all_cycles],
        'T_assoc_dhcp_ms': [c['T_assoc_dhcp_ms'] for c in all_cycles],
        'P_wake_deep_mean_mW': [c['P_wake_deep_mean_mW'] for c in all_cycles],
        'E_wake_light_mJ': [c['E_wake_light_mJ'] for c in all_cycles],
        'T_wake_light_ms': [c['T_wake_light_ms'] for c in all_cycles],
        'E_entry_light_mJ': [c['E_entry_light_mJ'] for c in all_cycles],
        'E_entry_deep_mJ': [c['E_entry_deep_mJ'] for c in all_cycles],
    }
    summary = {k: summarize(v, rng) for k, v in metrics.items()}
    result = {'description': 'Campaign 4: board-level sleep power (5 V input, INA238 ADCRANGE=1) and gross wake-up energy',
              'trim_ms': {'awake': TRIM_AWAKE_MS, 'sleep': TRIM_SLEEP_MS}, 'rise_threshold_mW': RISE_MW,
              'quality_gates': gates, 'all_passed': all(gates.values()), 'sessions': sessions,
              'summary': summary, 'cycles': all_cycles}

    print(f"cycles: {len(all_cycles)}  gates: {gates}")
    for k, s in summary.items():
        print(f"{k:22s} mean {s['mean']:9.3f}  sd {s['sd']:8.3f}  95% CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
              f"  range [{s['min']:.3f}, {s['max']:.3f}]")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(result, f, indent=2)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
