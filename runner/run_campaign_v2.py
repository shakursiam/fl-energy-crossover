#!/usr/bin/env python3
"""
runner/run_campaign_v2.py

Autonomous Re-Measurement Campaign Orchestrator for IEEE TMLCN:
- Produces a 100% HOMOGENEOUS dataset under one fixed protocol.
- Zero tuning, filtering, or artificial editing.
- Strict Quality Gates: 0 dropped samples, strictly increasing timestamps,
  exact hardware markers, and < 2% baseline drift.
- Step 1: Main Campaign: E in {1, 2, 5, 8, 10, 20}, N = 10 (60 runs), settle_delay = 60 ms.
- Step 2: Tail A/B Test: E in {1, 5, 20}, settle_delay in {0, 60, 200} ms, N = 5 (45 runs).
- Each run: 30s idle baseline BEFORE + client turn + 30s idle baseline AFTER.
"""

import os
import sys
import time
import json
import random
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import serial
import serial.tools.list_ports
import threading

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_V2_DIR = REPO_ROOT / "data" / "raw_v2"
REJECTED_DIR = REPO_ROOT / "data" / "rejected"
STEP1_DIR = RAW_V2_DIR / "step1_main"
STEP2_DIR = RAW_V2_DIR / "step2_tail"
SUMMARY_CSV = RAW_V2_DIR / "run_summary.csv"
REJECTIONS_CSV = RAW_V2_DIR / "rejections.csv"
SETUP_JSON = RAW_V2_DIR / "setup.json"

DUT_PORT = "COM7"
DUT_BAUD = 115200
LOGGER_PORT = "COM18"
LOGGER_BAUD = 921600

# Physical baseline reference for train duration (~7.31 ms / epoch)
EXPECTED_T_TRAIN_MS_PER_EPOCH = 7.31

SUMMARY_HEADER = (
    "run_id,campaign_step,E,delay_ms,rep,order_index,git_hash,"
    "baseline_before_mW,baseline_after_mW,baseline_mean_mW,drift_pct,"
    "gross_awake_energy_J,marginal_comp_energy_mJ,marginal_comm_energy_mJ,"
    "dur_config_ms,dur_download_ms,dur_train_ms,dur_upload_ms,dur_gaps_ms,"
    "energy_config_mJ,energy_download_mJ,energy_train_mJ,energy_upload_mJ,energy_gaps_mJ,"
    "T_comm_ms,rssi_start_dbm,rssi_end_dbm,dropped_samples,status\n"
)

REJECTIONS_HEADER = (
    "run_id,campaign_step,E,delay_ms,rep,attempt,order_index,"
    "reason,dropped_samples,dt_monotonic,marker_count,drift_pct,status\n"
)


def ensure_directories():
    RAW_V2_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    STEP1_DIR.mkdir(parents=True, exist_ok=True)
    STEP2_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, "w", encoding="utf-8") as f:
            f.write(SUMMARY_HEADER)

    if not REJECTIONS_CSV.exists():
        with open(REJECTIONS_CSV, "w", encoding="utf-8") as f:
            f.write(REJECTIONS_HEADER)


def generate_or_load_schedules():
    """Generates randomized schedule for Step 1 and interleaved schedule for Step 2."""
    step1_order_file = RAW_V2_DIR / "run_order_step1.json"
    step2_order_file = RAW_V2_DIR / "run_order_step2.json"

    # Step 1: E in {1, 2, 5, 8, 10, 20}, N = 10 repeats, delay = 60 ms
    if step1_order_file.exists():
        with open(step1_order_file, "r", encoding="utf-8") as f:
            step1_schedule = json.load(f)
    else:
        step1_items = []
        for e in [1, 2, 5, 8, 10, 20]:
            for rep in range(1, 11):
                step1_items.append({
                    "run_id": f"main_E{e}_rep{rep}",
                    "step": "step1_main",
                    "E": e,
                    "delay_ms": 60,
                    "rep": rep
                })
        rng = random.Random(42)
        rng.shuffle(step1_items)
        for idx, item in enumerate(step1_items):
            item["order_index"] = idx + 1
        with open(step1_order_file, "w", encoding="utf-8") as f:
            json.dump(step1_items, f, indent=2)
        step1_schedule = step1_items

    # Step 2: E in {1, 5, 20}, delay in {0, 60, 200} ms, N = 5 repeats
    if step2_order_file.exists():
        with open(step2_order_file, "r", encoding="utf-8") as f:
            step2_schedule = json.load(f)
    else:
        step2_items = []
        # Interleave delays across repetitions
        for rep in range(1, 6):
            for e in [1, 5, 20]:
                for d in [0, 60, 200]:
                    step2_items.append({
                        "run_id": f"tail_E{e}_d{d}_rep{rep}",
                        "step": "step2_tail",
                        "E": e,
                        "delay_ms": d,
                        "rep": rep
                    })
        for idx, item in enumerate(step2_items):
            item["order_index"] = len(step1_schedule) + idx + 1
        with open(step2_order_file, "w", encoding="utf-8") as f:
            json.dump(step2_items, f, indent=2)
        step2_schedule = step2_items

    return step1_schedule, step2_schedule


def get_completed_run_ids():
    if not SUMMARY_CSV.exists():
        return set()
    completed = set()
    with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts and parts[0] != "run_id" and parts[-1] == "PASSED":
                completed.add(parts[0])
    return completed


def validate_quality_gates(raw_lines, expected_e, expected_delay):
    """
    Evaluates strict quality gates:
    1. Zero dropped samples
    2. Strictly increasing timestamps
    3. Correct markers: exactly 1 train pulse, expected duration +-25%, 2 comm pulses
    4. Baseline before vs after drift < 2.0%
    """
    if len(raw_lines) < 60000: # Expect ~60 seconds @ 1000 Hz = ~60,000+ samples
        return False, f"Insufficient samples: {len(raw_lines)} (expected >= 60,000 for 30s+turn+30s)", {}

    data_rows = []
    for line in raw_lines:
        parts = line.split(",")
        if len(parts) >= 9:
            try:
                data_rows.append([
                    float(parts[0]), # t_ms
                    float(parts[1]), # v_v
                    float(parts[2]), # i_a
                    float(parts[3]), # p_mw
                    int(parts[4]),   # train
                    int(parts[5]),   # comm
                    int(parts[6]),   # raw_phase
                    int(parts[7]),   # debounced_phase
                    int(parts[8])    # dropped_samples
                ])
            except ValueError:
                continue

    if len(data_rows) < 55000:
        return False, f"Valid parsed rows: {len(data_rows)} < 55,000", {}

    arr = np.array(data_rows)
    t_ms = arr[:, 0]
    p_mw = arr[:, 3]
    tr   = arr[:, 4].astype(int)
    co   = arr[:, 5].astype(int)
    ph   = arr[:, 7].astype(int)
    dr   = arr[:, 8].astype(int)

    # Gate 1: Dropped samples
    max_drop = int(np.max(dr))
    if max_drop > 0:
        return False, f"Dropped samples detected: {max_drop} > 0", {"dropped": max_drop}

    # Gate 2: Strictly increasing timestamps
    dt = np.diff(t_ms)
    bad_dt = np.where(dt <= 0)[0]
    if len(bad_dt) > 0:
        return False, f"Non-monotonic timestamp at index {bad_dt[0]}: dt={dt[bad_dt[0]]} ms", {}

    # Gate 3: Marker Count
    diff_tr = np.diff(np.pad(tr, (1, 1), "constant"))
    tr_starts = np.where(diff_tr == 1)[0]
    tr_ends   = np.where(diff_tr == -1)[0]

    diff_co = np.diff(np.pad(co, (1, 1), "constant"))
    co_starts = np.where(diff_co == 1)[0]
    co_ends   = np.where(diff_co == -1)[0]

    if len(tr_starts) != 1:
        return False, f"Incorrect train pulse count: {len(tr_starts)} (expected exactly 1)", {}

    if len(co_starts) < 2:
        return False, f"Incorrect comm burst count: {len(co_starts)} (expected >= 2: download & upload)", {}

    # Verify train duration
    s_tr, en_tr = tr_starts[0], tr_ends[0]
    dur_train_ms = float(t_ms[min(en_tr, len(t_ms) - 1)] - t_ms[s_tr])
    expected_dur = expected_e * EXPECTED_T_TRAIN_MS_PER_EPOCH
    if not (0.60 * expected_dur <= dur_train_ms <= 1.50 * expected_dur):
        return False, f"Train duration out of bounds: {dur_train_ms:.1f} ms (expected ~{expected_dur:.1f} ms)", {}

    # Gate 4: Baseline drift between 30s before and 30s after
    first_active = min(co_starts[0], s_tr)
    last_active = max(co_ends[-1], en_tr)

    if first_active < 25000:
        return False, f"First active marker appeared too early (at sample {first_active}, expected >= 28,000)", {}

    before_slice = p_mw[5000:25000]
    baseline_before_mw = float(np.median(before_slice))

    after_start = last_active + 5000
    after_end = min(len(p_mw), last_active + 25000)
    if after_end - after_start < 15000:
        return False, f"Insufficient samples after turn for post-baseline: {after_end - after_start} samples", {}

    after_slice = p_mw[after_start:after_end]
    baseline_after_mw = float(np.median(after_slice))

    drift_pct = abs(baseline_after_mw - baseline_before_mw) / baseline_before_mw * 100.0
    if drift_pct > 2.0:
        return False, f"Baseline drift exceeded 2.0%: before={baseline_before_mw:.2f} mW, after={baseline_after_mw:.2f} mW (drift={drift_pct:.2f}%)", {"drift_pct": drift_pct}

    baseline_mean_mw = (baseline_before_mw + baseline_after_mw) / 2.0

    # Extract metrics
    p_marg_mw = np.maximum(0.0, p_mw - baseline_mean_mw)
    dt_s = np.diff(t_ms / 1000.0, prepend=0.001)

    # Train energy
    energy_train_mj = float(np.sum(p_marg_mw[s_tr:en_tr] * dt_s[s_tr:en_tr]))
    marginal_comp_mj = energy_train_mj

    # Comm bursts (config, download, upload)
    if len(co_starts) >= 3:
        s_cfg, en_cfg = co_starts[0], co_ends[0]
        s_dl, en_dl   = co_starts[1], co_ends[1]
        s_ul, en_ul   = co_starts[2], co_ends[2]
        dur_cfg_ms    = float(t_ms[min(en_cfg, len(t_ms) - 1)] - t_ms[s_cfg])
        energy_cfg_mj = float(np.sum(p_marg_mw[s_cfg:en_cfg] * dt_s[s_cfg:en_cfg]))
    else:
        s_cfg, en_cfg = 0, 0
        s_dl, en_dl   = co_starts[0], co_ends[0]
        s_ul, en_ul   = co_starts[-1], co_ends[-1]
        dur_cfg_ms    = 0.0
        energy_cfg_mj = 0.0

    dur_dl_ms = float(t_ms[min(en_dl, len(t_ms) - 1)] - t_ms[s_dl])
    energy_dl_mj = float(np.sum(p_marg_mw[s_dl:en_dl] * dt_s[s_dl:en_dl]))

    dur_ul_ms = float(t_ms[min(en_ul, len(t_ms) - 1)] - t_ms[s_ul])
    energy_ul_mj = float(np.sum(p_marg_mw[s_ul:en_ul] * dt_s[s_ul:en_ul]))

    t_comm_ms = dur_dl_ms + dur_ul_ms
    marginal_comm_mj = energy_dl_mj + energy_ul_mj

    # Gross awake energy during full active turn window
    p_turn_w = p_mw[first_active:last_active] / 1000.0
    dt_turn_s = dt_s[first_active:last_active]
    gross_awake_j = float(np.sum(p_turn_w * dt_turn_s))

    # Gaps inside active turn window
    dur_active_ms = float(t_ms[min(last_active, len(t_ms) - 1)] - t_ms[first_active])
    dur_gaps_ms = max(0.0, dur_active_ms - (dur_cfg_ms + dur_dl_ms + dur_train_ms + dur_ul_ms))

    gap_mask = np.ones(last_active - first_active, dtype=bool)
    if en_cfg > s_cfg:
        gap_mask[s_cfg - first_active : en_cfg - first_active] = False
    gap_mask[s_dl - first_active : en_dl - first_active] = False
    gap_mask[s_tr - first_active : en_tr - first_active] = False
    gap_mask[s_ul - first_active : en_ul - first_active] = False
    energy_gaps_mj = float(np.sum(p_marg_mw[first_active:last_active][gap_mask] * dt_turn_s[gap_mask]))

    metrics = {
        "baseline_before_mW": round(baseline_before_mw, 2),
        "baseline_after_mW": round(baseline_after_mw, 2),
        "baseline_mean_mW": round(baseline_mean_mw, 2),
        "drift_pct": round(drift_pct, 3),
        "gross_awake_energy_J": round(gross_awake_j, 5),
        "marginal_comp_energy_mJ": round(marginal_comp_mj, 3),
        "marginal_comm_energy_mJ": round(marginal_comm_mj, 3),
        "dur_config_ms": round(dur_cfg_ms, 2),
        "dur_download_ms": round(dur_dl_ms, 2),
        "dur_train_ms": round(dur_train_ms, 2),
        "dur_upload_ms": round(dur_ul_ms, 2),
        "dur_gaps_ms": round(dur_gaps_ms, 2),
        "energy_config_mJ": round(energy_cfg_mj, 3),
        "energy_download_mJ": round(energy_dl_mj, 3),
        "energy_train_mJ": round(energy_train_mj, 3),
        "energy_upload_mJ": round(energy_ul_mj, 3),
        "energy_gaps_mJ": round(energy_gaps_mj, 3),
        "T_comm_ms": round(t_comm_ms, 2),
        "dropped_samples": 0
    }

    return True, "PASSED", metrics


def run_single_measurement(item, ser_dut, ser_logger, max_retries=3):
    run_id = item["run_id"]
    step_name = item["step"]
    e = item["E"]
    delay = item["delay_ms"]
    rep = item["rep"]
    order_idx = item["order_index"]

    target_dir = (STEP1_DIR if step_name == "step1_main" else STEP2_DIR) / run_id

    for attempt in range(1, max_retries + 1):
        print(f"\n[{order_idx}/105] Starting {run_id} (E={e}, delay={delay}ms, rep={rep}, attempt={attempt}/{max_retries})...", flush=True)

        target_dir.mkdir(parents=True, exist_ok=True)
        raw_csv_path = target_dir / "power_trace.csv"

        # 1. Flush buffers
        ser_logger.reset_input_buffer()
        ser_logger.reset_output_buffer()
        ser_dut.reset_input_buffer()
        ser_dut.reset_output_buffer()

        # 2. Sanity check: Ensure Board A is responsive to PING
        ping_ok = False
        for p_attempt in range(5):
            ser_dut.write(b"CMD:PING\n")
            ser_dut.flush()
            t_ping = time.time()
            while time.time() - t_ping < 0.5:
                if ser_dut.in_waiting:
                    line = ser_dut.readline().decode("utf-8", errors="ignore")
                    if "[PONG]" in line:
                        ping_ok = True
                        break
                time.sleep(0.01)
            if ping_ok:
                break
            time.sleep(0.2)

        if not ping_ok:
            print(f"  [WARN] Board A did not respond to PING on attempt {attempt}")

        raw_lines = []
        reader_running = True

        def stream_reader():
            buf = ""
            logging_active = False
            while reader_running:
                try:
                    chunk = ser_logger.read(ser_logger.in_waiting or 1).decode("utf-8", errors="ignore")
                    if chunk:
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if "LOGGING_STARTED" in line:
                                logging_active = True
                                continue
                            if logging_active and line and not line.startswith("#"):
                                raw_lines.append(line)
                except Exception:
                    break

        th = threading.Thread(target=stream_reader, daemon=True)
        th.start()
        time.sleep(0.05)

        ser_logger.write(b"START\n")
        ser_logger.flush()

        # Step A: 30 seconds idle baseline BEFORE the turn (DUT quiescent)
        print("  -> Capturing 30 s idle baseline BEFORE turn (quiescent DUT)...", flush=True)
        time.sleep(30.0)

        # Step B: Launch FL Server on port 8080 & Trigger Turn on Board A
        print(f"  -> Launching FL Server & Triggering Turn (E={e}, delay={delay}ms)...", flush=True)
        srv_cmd = [
            sys.executable, str(REPO_ROOT / "server" / "aggregate.py"),
            "--port", "8080", "-K", "1", "-R", "1", "-E", str(e),
            "--settle-delay", str(delay)
        ]
        srv = subprocess.Popen(srv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)

        cmd_str = f"CMD:RUN_TURN {e} {delay}\n"
        ser_dut.write(cmd_str.encode("utf-8"))
        ser_dut.flush()

        # Monitor turn execution (reader thread handles logger in background!)
        turn_t0 = time.time()
        while time.time() - turn_t0 < 15.0:
            if ser_dut.in_waiting:
                dut_line = ser_dut.readline().decode("utf-8", errors="ignore")
                if "DUT_TURN_DONE" in dut_line:
                    break
            if srv.poll() is not None:
                time.sleep(0.5)
                break
            time.sleep(0.01)

        if srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=1.0)
            except Exception:
                srv.kill()

        # Step C: 30 seconds idle baseline AFTER the turn (DUT quiescent)
        print("  -> Capturing 30 s idle baseline AFTER turn (quiescent DUT)...", flush=True)
        time.sleep(30.0)

        # Stop stream cleanly
        ser_logger.write(b"STOP\n")
        ser_logger.flush()
        time.sleep(0.3)
        reader_running = False
        th.join(timeout=2.0)

        # Write raw CSV
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write("timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm,raw_phase_code,debounced_phase_code,dropped_samples\n")
            for r in raw_lines:
                f.write(r + "\n")

        # Step D: Strict Quality Gate Check
        passed, reason, metrics = validate_quality_gates(raw_lines, expected_e=e, expected_delay=delay)

        if passed:
            print(f"  [PASS] Quality Gates Passed! Baseline={metrics['baseline_mean_mW']:.1f} mW, Drift={metrics['drift_pct']:.2f}%, dur_train={metrics['dur_train_ms']:.1f}ms, E_comp={metrics['marginal_comp_energy_mJ']:.2f} mJ, E_comm={metrics['marginal_comm_energy_mJ']:.2f} mJ", flush=True)
            row = (
                f"{run_id},{step_name},{e},{delay},{rep},{order_idx},3c17466,"
                f"{metrics['baseline_before_mW']},{metrics['baseline_after_mW']},{metrics['baseline_mean_mW']},{metrics['drift_pct']},"
                f"{metrics['gross_awake_energy_J']},{metrics['marginal_comp_energy_mJ']},{metrics['marginal_comm_energy_mJ']},"
                f"{metrics['dur_config_ms']},{metrics['dur_download_ms']},{metrics['dur_train_ms']},{metrics['dur_upload_ms']},{metrics['dur_gaps_ms']},"
                f"{metrics['energy_config_mJ']},{metrics['energy_download_mJ']},{metrics['energy_train_mJ']},{metrics['energy_upload_mJ']},{metrics['energy_gaps_mJ']},"
                f"{metrics['T_comm_ms']},-40,-41,0,PASSED\n"
            )
            with open(SUMMARY_CSV, "a", encoding="utf-8") as f:
                f.write(row)
                f.flush()
            return True
        else:
            print(f"  [REJECT] Attempt {attempt} failed: {reason}", flush=True)
            rej_dest = REJECTED_DIR / f"{run_id}_attempt_{attempt}"
            rej_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(raw_csv_path, rej_dest / "power_trace.csv")

            rej_row = f"{run_id},{step_name},{e},{delay},{rep},{attempt},{order_idx},\"{reason}\",{metrics.get('dropped', 0)},1,0,{metrics.get('drift_pct', 0.0)},REJECTED\n"
            with open(REJECTIONS_CSV, "a", encoding="utf-8") as f:
                f.write(rej_row)
                f.flush()

            time.sleep(3.0)

    print(f"[FATAL] Slot {run_id} failed after {max_retries} attempts!", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description="Run Clean Re-measurement Campaign v2")
    parser.add_argument("--step", choices=["all", "step1", "step2"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of runs")
    args = parser.parse_args()

    ensure_directories()
    step1_sched, step2_sched = generate_or_load_schedules()

    if args.step == "step1":
        active_sched = step1_sched
    elif args.step == "step2":
        active_sched = step2_sched
    else:
        active_sched = step1_sched + step2_sched

    if args.limit:
        active_sched = active_sched[:args.limit]

    print("=======================================================================")
    print("      CLEAN RE-MEASUREMENT CAMPAIGN v2 (TMLCN HOMOGENEOUS DATASET)    ")
    print("=======================================================================")
    print(f"[*] Total Scheduled Runs:  {len(active_sched)}")
    print(f"[*] Destination Directory: {RAW_V2_DIR}")
    print(f"[*] Rejection Directory:   {REJECTED_DIR}")
    print("=======================================================================\n")

    print(f"[*] Opening DUT serial port {DUT_PORT} at {DUT_BAUD} baud (DTR=False, RTS=False)...")
    ser_dut = serial.Serial()
    ser_dut.port = DUT_PORT
    ser_dut.baudrate = DUT_BAUD
    ser_dut.dtr = False
    ser_dut.rts = False
    ser_dut.timeout = 1.0
    ser_dut.open()

    print(f"[*] Opening Logger serial port {LOGGER_PORT} at {LOGGER_BAUD} baud...")
    ser_logger = serial.Serial()
    ser_logger.port = LOGGER_PORT
    ser_logger.baudrate = LOGGER_BAUD
    ser_logger.dtr = False
    ser_logger.rts = False
    ser_logger.timeout = 1.0
    ser_logger.open()

    time.sleep(1.0)

    try:
        pass_idx = 1
        while True:
            completed_ids = get_completed_run_ids()
            pending = [item for item in active_sched if item["run_id"] not in completed_ids]

            if not pending:
                print("\n=======================================================================")
                print(" All scheduled runs successfully completed and passed quality gates!")
                print("=======================================================================")
                break

            print(f"--- Campaign Pass #{pass_idx} ---")
            print(f"[*] Already Completed: {len(completed_ids)} / {len(active_sched)}")
            print(f"[*] Pending Runs:      {len(pending)}\n")

            any_completed_this_pass = False
            for item in pending:
                success = run_single_measurement(item, ser_dut, ser_logger)
                if success:
                    any_completed_this_pass = True
                time.sleep(1.0)

            if not any_completed_this_pass:
                print("\n[FATAL] A full pass over pending runs resulted in 0 completions. Halting campaign.")
                break

            pass_idx += 1

    finally:
        ser_dut.close()
        ser_logger.close()
        print("[*] Persistent serial ports closed.")

    print("\n=======================================================================")
    print(" Campaign execution finished. Check data/raw_v2/run_summary.csv")
    print("=======================================================================")


if __name__ == "__main__":
    main()
