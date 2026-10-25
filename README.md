# Energy Allocation Between Computation and Wi-Fi Transmission in TinyML Federated Learning

Code and data for the manuscript

> Md Abdus Shakur Siam and Farag Azzedin, "Energy Allocation Between Computation and Wi-Fi Transmission in TinyML Federated Learning: A Measured Crossover," submitted to *IEEE Transactions on Machine Learning in Communications and Networking*, 2026.

An ESP32-S3 (Board A) trains the 780-parameter classification head of a frozen keyword-spotting network and exchanges its update (3,120 B) with a FedAvg server over 2.4 GHz Wi-Fi. A TI INA238 on the 5 V supply of Board A is sampled at 1 kHz by a second ESP32-S3 (Board B), which also records the GPIO phase markers of Board A. The repository contains the 105 raw power traces, the firmware of both boards, the host scripts that ran the campaign, and the analysis scripts that produce every number and figure of the paper.

## Contents

```
data/raw_v2/          105 raw power traces (1 kHz) and campaign metadata
  step1_main/         60 runs: E in {1, 2, 5, 8, 10, 20}, 10 repetitions, randomized order, 60 ms settling delay
  step2_tail/         45 runs: E in {1, 5, 20} x settling delay {0, 60, 200} ms x 5 repetitions
  setup.json          instrument, board and RF configuration
  run_summary.csv     per-run phase times and energies
  rejections.csv      rejected attempts and the reason for each
  run_order_*.json    acquisition order
  sha256_manifest.txt checksums of the traces
data/processed/       backbone embeddings used by the convergence simulation
data_summary/         outputs of the analysis scripts (JSON) and the per-run tables
analysis/             analysis and figure scripts (see below)
firmware/BOARD_A/     device under test: local training, HTTP exchange, GPIO phase markers (Arduino)
firmware/BOARD_B/     logger: INA238 at 1 kHz plus marker capture, streamed over USB (Arduino)
server/aggregate.py   FedAvg server (HTTP)
runner/               campaign orchestration (randomized order, quality gates, retries)
logger/               host-side capture of the Board B stream
reproduce_all.py      regenerates all numbers and figures
```

## Reproducing the results

Python 3.9 or later:

```
pip install -r requirements.txt
python reproduce_all.py                 # numbers and figures from the raw traces
python reproduce_all.py --convergence   # also reruns the FL convergence simulation
```

Outputs are written to `data_summary/` and `figures/`. The bootstrap intervals use fixed seeds and reproduce the values in the paper.

| Script | Produces |
|---|---|
| `verify_campaign_v2.py` | trace checks (dropped samples, timestamps, markers, drift) and recomputed per-run energies |
| `make_unclamped_summary.py` | `run_summary.csv` from the raw traces |
| `stats_campaign_v2.py` | regression fits and hypothesis tests (`stats.json`) |
| `paper_numbers_v2.py` | fits, crossover estimates and bootstrap intervals (`paper_numbers.json`) |
| `tail_profile_v2.py` | post-communication tail profiles (`tail_profile.json`) |
| `analytical_baselines.py` | Shannon, switching-energy and datasheet models of Section IV |
| `derived_v2.py` | sleep and contention analysis, model mismatch (`derived.json`) |
| `convergence_v3.py` | FL convergence simulation and rounds to target (`convergence.json`) |
| `convergence_robustness.py` | optimum local epochs for other targets and wake-up energies |
| `make_figs_v2.py`, `make_figs_v3.py`, `plot_fig2_new_logger.py` | figures |

## Raw trace format

Each `power_trace.csv` has one row per millisecond:

| Column | Unit | Description |
|---|---|---|
| `timestamp_ms` | ms | time since the logger started |
| `voltage_V` | V | bus voltage at Board A |
| `current_A` | A | supply current of Board A (15 mΩ shunt) |
| `power_mW` | mW | bus voltage times current |
| `gpio_train` | 0/1 | training marker (Board A GPIO 10) |
| `gpio_comm` | 0/1 | communication marker (Board A GPIO 11) |
| `raw_phase_code` | 0-7 | 3-bit phase bus (GPIO 12-14) as sampled |
| `debounced_phase_code` | 0-7 | phase bus after one-sample confirmation |
| `dropped_samples` | count | samples lost by the logger (0 in all accepted runs) |

The RSSI columns of `run_summary.csv` hold the values read at the start and end of the campaign; RSSI was not logged per run.

## Running the testbed

1. Set `WIFI_SSID`, `WIFI_PASS` and `SERVER_IP` at the top of `firmware/BOARD_A/BOARD_A.ino` and flash both boards with the Arduino ESP32 core.
2. Wire Board A GPIO 10-14 to Board B GPIO 6, 7, 8, 9 and 18, each with a 100 kΩ pull-down, and place the INA238 shunt on the 5 V input of Board A. Disconnect the USB power of Board A during measurement.
3. Set the serial ports at the top of `runner/run_campaign_v2.py` and run it; it starts `server/aggregate.py` and records one trace per run.

## License

Code and firmware: MIT. Data: CC BY 4.0. See `LICENSE`.
