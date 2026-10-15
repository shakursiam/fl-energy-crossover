# Empirical Energy Crossover in TinyML Federated Learning: Replication Package and Raw Power Measurement Traces

[![IEEE TMLCN](https://img.shields.io/badge/IEEE_Transactions-TMLCN_2026-blue.svg)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Traces Verified](https://img.shields.io/badge/Raw_Traces-105%20Verified%20(100%25)-success.svg)]()
[![Sampling Rate](https://img.shields.io/badge/Sampling_Rate-1000_Hz_INA238-orange.svg)]()

This repository contains the complete open-science replication package, hardware measurement firmware, analysis scripts, processed datasets, publication figures, and 105 raw physical silicon power traces for the manuscript:

> **Md Abdus Shakur Siam and Farag Azzedin**, *"Energy Allocation Between Computation and Wi-Fi Transmission in TinyML Federated Learning: A Measured Crossover,"* under review at **IEEE Transactions on Machine Learning in Communications and Networking (TMLCN)**, 2026.

---

## Data Availability Statement

In accordance with IEEE open science and data availability standards:
> *"The complete dataset of 105 raw silicon power measurement traces, analysis scripts, firmware source code, and manuscript source files are publicly available in this repository: [https://github.com/shakursiam/fl-energy-crossover](https://github.com/shakursiam/fl-energy-crossover)."*

---

## Executive Summary of Findings

Most wireless federated learning (FL) algorithms operate on the foundational premise that communication over the radio dominates energy expenditure, while on-device computation is negligible. We experimentally test this premise on an **Espressif ESP32-S3** microcontroller training on-device keyword-spotting models ($3{,}120$-byte FP32 updates) over **2.4 GHz 802.11b/g/n Wi-Fi** sampled continuously at **1 kHz** with an independent **Texas Instruments INA238** power monitor.

```
+------------------------------------------------------------------------------------------------+
| MAIN PHYSICAL CROSSOVER FINDINGS (105 SILICON TRACES)                                          |
+------------------------------------------------------------------------------------------------+
|  1. Computation Scaling:    Delta E_comp(E) = 1.002 * E - 0.010 mJ  (R^2 = 0.999, c_0 ~ 0)     |
|  2. Marginal Communication: Delta E_comm = 9.748 mJ per turn (flat across epochs)              |
|  3. Empirical Crossover:    E* = 9.74 epochs (95% CI: [8.79, 10.85] epochs in main campaign)   |
|  4. Session Variation:      E* = 7.42 epochs in session 2 (RF condition sensitivity: 7.4 - 9.7) |
|  5. Quiescent Power Floor:  P_base = 561.1 mW awake baseline power draw                        |
|  6. Radio Tail Relaxation:  0.506 mJ absorbed by computation if training starts within 60 ms   |
|  7. Analytical Model Error: Standard Shannon and switching models miss physical energy by >89%|
+------------------------------------------------------------------------------------------------+
```

---

## Repository Structure

```
.
├── README.md                      # This comprehensive guide and data dictionary
├── LICENSE                        # MIT License (Code) and CC-BY-4.0 (Data/Figures)
├── requirements.txt               # Scientific Python dependencies
├── reproduce_all.py               # Automated one-click replication pipeline
├── data/
│   ├── raw_v2/                    # 105 raw physical silicon 1 kHz power traces
│   │   ├── setup.json             # Instrument calibration, shunt resistance, ADC config
│   │   ├── rejections.csv         # Audit log of rejected runs and rejection criteria
│   │   ├── step1_main/            # 60 main campaign runs (E in {1,2,5,8,10,20} x 10 runs)
│   │   └── step2_tail/            # 45 radio tail relaxation runs (delays 0, 50, 100, 200, 500 ms)
│   └── data_summary/
│       ├── run_summary.csv        # Primary unclamped dataset matching main.tex
│       ├── run_summary_pipeline_clamped.csv  # Clamped dataset for pipeline auditing
│       ├── paper_numbers.json     # All statistical fit numbers and bootstrap CIs
│       ├── stats.json             # Aggregated stats, OLS fits, hypothesis tests
│       ├── derived.json           # Sleep, contention, and convergence metrics
│       ├── tail_profile.json      # Binned tail relaxation decay profiles
│       ├── recomputed_runs.csv    # Recomputed metrics from all 105 raw traces
│       └── gap_profiles.json      # Raw relaxation gap profiles
├── analysis/
│   ├── verify_campaign_v2.py      # Independent trace verification & phase reintegration
│   ├── make_unclamped_summary.py  # Unclamped integration across communication windows
│   ├── paper_numbers_v2.py        # Manuscript number generator and bootstrap estimator
│   ├── stats_campaign_v2.py       # Full statistical analysis and hypothesis tests
│   ├── derived_v2.py              # Evaluates analytical sleep, contention, & convergence
│   ├── tail_profile_v2.py         # Profiles the post-communication radio tail decay
│   ├── make_figs_v2.py            # Generates Figs 3, 4, 5, 6, 7 (PDF + PNG)
│   └── plot_fig2_new_logger.py    # Generates Fig 2 (1 kHz time-series trace, PDF + PNG)
├── figures/                       # All 7 publication figures (PDF vector & high-res PNG)
│   ├── fig1_power_time_series.pdf
│   ├── fig1_power_time_series.png
│   ├── fig_crossover_v2.pdf
│   ├── fig_crossover_v2.png
│   ├── fig_tail_ab.pdf
│   ├── fig_tail_ab.png
│   ├── fig_tail_profile.pdf
│   ├── fig_tail_profile.png
│   ├── fig_comm_variation.pdf
│   ├── fig_comm_variation.png
│   ├── fig_energy_to_target_v2.pdf
│   ├── fig_energy_to_target_v2.png
│   └── pipeline.drawio.pdf
├── firmware/
│   ├── Board_A_DUT/               # TinyML client training head (ESP32-S3)
│   │   └── Board_A_DUT.ino
│   └── Board_B_Logger/            # 1 kHz INA238 hardware power logger
│       └── Board_B_Logger.ino
└── manuscript/
    ├── main.tex                   # Complete LaTeX manuscript (IEEEtran class)
    ├── main.pdf                   # Compiled publication-ready PDF (7 pages)
    ├── refs.bib                   # BibTeX bibliography
    ├── IEEEtran.cls               # IEEE Transactions LaTeX class
    ├── IEEEtran.bst               # IEEE Transactions BibTeX style
    ├── cover_letter.tex           # Submission cover letter source
    ├── cover_letter.pdf           # Compiled cover letter
    └── CAMPAIGN_V2_REPORT.md      # Comprehensive methodology and calibration report
```

---

## Hardware Measurement Architecture

```
                    +--------------------------------------------------+
                    |           5.00V Low-Noise Supply Rail            |
                    +-------------------------+------------------------+
                                              |
                                     +--------v--------+
                                     |  100 mOhm Shunt  |
                                     +---+-----------+-+
                                         |           |
            +----------------------------v-+       +-v-----------------------------+
            |      TI INA238 Power Monitor |       | Seeed XIAO ESP32-S3 Sense     |
            |      (16-bit ADC, 1000 Hz)   |       | Device Under Test (DUT)       |
            +--------------+---------------+       +---------------+---------------+
                           | I2C (400 kHz)                         | GPIO Markers
            +--------------v---------------+                       | (Microsecond-
            | Seeed XIAO ESP32-S3 Logger   |<----------------------+  level sync)
            | (Board B, Continuous Stream) |
            +--------------+---------------+
                           | USB-CDC (Serial Stream)
            +--------------v---------------+
            | Workstation Host Logger      |
            +------------------------------+
```

### Bill of Materials (BOM)
1. **Device Under Test (DUT, Board A)**: Seeed Studio XIAO ESP32-S3 Sense (Dual-core Xtensa LX7 @ 240 MHz, 8 MB PSRAM, 8 MB Flash, 2.4 GHz Wi-Fi).
2. **Hardware Logger (Board B)**: Seeed Studio XIAO ESP32-S3.
3. **Power Monitor**: Texas Instruments INA238 (16-bit ADC, 1000 conversions/sec continuous mode, high-side sensing).
4. **Current Shunt**: $100	ext{ m}\Omega$ precision current sensing resistor.
5. **Phase Delimitation**: Dedicated digital GPIO pins (GPIO 1, 2, 3) toggling at phase transitions:
   - Code `0`: Quiescent baseline / idle
   - Code `1`: Config request / reply
   - Code `2`: Model download handshake
   - Code `3`: Model download payload
   - Code `4`: Model upload handshake
   - Code `5`: Model upload payload
   - Code `6`: On-device SGD mini-batch training

---

## Reproduction Guide

### Prerequisites
Python 3.8+ with standard scientific libraries:
```bash
pip install -r requirements.txt
```

### One-Click End-to-End Reproduction
Execute the automated master runner:
```bash
python reproduce_all.py
```
This script will:
1. Re-analyze all 105 raw silicon traces in `data/raw_v2/`
2. Validate monotonic timestamps, zero dropped samples, and baseline drift (<0.84%)
3. Re-fit OLS lines and execute 5,000 bootstrap resamplings
4. Re-generate all 7 publication figures into `figures/`
5. Print a verification scorecard verifying exact concordance with `main.tex`.

---

## Data Dictionary

### 1. `power_trace.csv` (105 Raw Trace Files)
| Column Name | Units | Description |
|---|---|---|
| `timestamp_ms` | ms | Monotonic elapsed time since logger initialization (1.0 ms intervals). |
| `current_mA` | mA | High-side current measured across the 100 mΩ shunt by INA238. |
| `bus_voltage_V` | V | Regulated supply rail voltage (nominal 5.00 V). |
| `power_mW` | mW | Instantaneous electrical power ($V_{m bus} 	imes I_{m shunt}$). |
| `gpio_comm` | binary | Physical GPIO marker: 1 when Wi-Fi radio is actively transmitting/receiving. |
| `gpio_train` | binary | Physical GPIO marker: 1 when MCU CPU executes SGD backpropagation. |
| `raw_phase_code` | integer | Phase code (0: baseline, 1-5: comm phases, 6: training). |
| `dropped_samples` | count | Dropped sample counter from hardware I2C buffer (0 across all runs). |

### 2. `run_summary.csv` (Authoritative Dataset)
| Column Name | Units | Description |
|---|---|---|
| `run_id` | string | Unique run identifier (e.g., `main_E5_rep3`, `tail_d60_rep2`). |
| `campaign_step` | string | `step1_main` (60 runs) or `step2_tail` (45 runs). |
| `E` | count | Local SGD training epochs per client turn ($E \in \{1, 2, 5, 8, 10, 20\}$). |
| `delay_ms` | ms | Post-communication settling delay before training starts ($0, 50, 100, 200, 500$). |
| `rep` | index | Repetition index (0 to 9). |
| `marginal_comp_energy_mJ` | mJ | Energy consumed during training above quiescent baseline floor. |
| `marginal_comm_energy_mJ` | mJ | Unclamped energy consumed during Wi-Fi exchange above baseline. |
| `dur_train_ms` | ms | Duration of SGD training phase. |
| `T_comm_ms` | ms | Total airtime duration of Wi-Fi communication phases. |
| `baseline_mean_mW` | mW | Quiescent baseline floor power ($P_{m base} pprox 561.1$ mW). |
| `drift_pct` | % | Relative baseline drift between pre-turn and post-turn windows. |

---

## Technical Note: Unclamped vs. Clamped Baseline Accounting

In Appendix A of the manuscript, we explicitly document the accounting difference between pipeline-clamped and unclamped marginal communication energies:
- **Pipeline Clamping**: The instrumentation pipeline originally computed $\int \max(0, P(t) - P_{m base})\,dt$ to avoid integrating noise excursions during high-frequency switching. This produced $\Delta E_{m comm} = 10.157	ext{ mJ}$.
- **Unclamped Accounting**: Re-integrating the exact physical integral $\int (P(t) - P_{m base})\,dt$ accounts for negative transient noise swings around baseline, yielding $\Delta E_{m comm} = 9.748	ext{ mJ}$ ($E^* = 9.74	ext{ epochs}$).
- For complete auditability, both datasets are provided (`run_summary.csv` for unclamped, `run_summary_pipeline_clamped.csv` for clamped). Recomputation concordance holds within $0.002	ext{ mJ}$ across 101 of 102 validated runs.

---

## Citation

If you use this dataset, measurement methodology, firmware, or codebase in your research, please cite:

```bibtex
@article{siam2026energy,
  author    = {Siam, Md Abdus Shakur and Azzedin, Farag},
  title     = {Energy Allocation Between Computation and {Wi-Fi} Transmission in {TinyML} Federated Learning: A Measured Crossover},
  journal   = {IEEE Transactions on Machine Learning in Communications and Networking},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

- **Source Code and Firmware**: [MIT License](LICENSE)
- **Silicon Traces, Datasets, and Figures**: [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
