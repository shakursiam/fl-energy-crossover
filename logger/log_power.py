#!/usr/bin/env python3
"""
logger/log_power.py

Host-side energy logging orchestrator:
1. Connects to the ESP32-S3 energy-logging host via serial.
2. Captures INA238 power samples and GPIO phase markers (train, comm) at ~45 Hz.
3. Automatically writes to data/raw/<E>/<repetition>/power_trace.csv with header:
   timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm
4. Automatically tracks completed FL rounds (via gpio_comm falling edge transitions
   or optional DUT serial log monitoring) or terminates after a wall-clock timeout.
5. Cleanly flushes and closes the CSV file.
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
import serial
import serial.tools.list_ports

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_HEADER_6 = "timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm"
CSV_HEADER_9 = "timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm,raw_phase_code,debounced_phase_code,dropped_samples"


def list_available_ports():
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def main():
    parser = argparse.ArgumentParser(description="Log INA238 power traces and GPIO phase markers from logging host.")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port of the ESP32-S3 logging host (e.g. COM3 or /dev/ttyUSB0).")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="Local epochs E for this run (e.g. 1, 5, 10, 20).")
    parser.add_argument("-r", "--repetition", type=int, default=1,
                        help="Repetition index for this run (e.g. 1, 2, 3).")
    parser.add_argument("-n", "--rounds", type=int, default=10,
                        help="Number of FL rounds to capture before automatically stopping.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional wall-clock duration limit (seconds). If set, stops after this duration.")
    parser.add_argument("--dut-port", type=str, default=None,
                        help="Optional serial port of DUT ESP32-S3 to capture DUT console output concurrently.")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate for serial connection (default: 115200).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Custom output directory. Defaults to data/raw/<E>/<repetition>/")
    args = parser.parse_args()

    # Determine destination directory & file
    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = REPO_ROOT / "data" / "raw" / f"E_{args.epochs}" / f"rep_{args.repetition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = run_dir / "power_trace.csv"
    dut_log_path = run_dir / "dut_serial.log"
    meta_path = run_dir / "meta.json"

    # Select serial port if not provided
    port = args.port
    if not port:
        avail = list_available_ports()
        if not avail:
            print("[ERROR] No serial ports found on the system!")
            sys.exit(1)
        print(f"[?] Available serial ports: {avail}")
        port = avail[0]
        print(f"[*] No --port specified. Defaulting to first available port: {port}")

    print("===============================================================")
    print("           ESP32-S3 / INA238 Power Logging Host               ")
    print("===============================================================")
    print(f"[*] Logger Port:      {port} @ {args.baud} baud")
    print(f"[*] Local Epochs (E): {args.epochs}")
    print(f"[*] Repetition:       {args.repetition}")
    print(f"[*] Target Rounds:    {args.rounds}")
    print(f"[*] Output CSV:       {csv_file_path}")
    if args.dut_port:
        print(f"[*] DUT Serial Port:  {args.dut_port}")
    if args.duration:
        print(f"[*] Max Duration:     {args.duration} s")
    print("===============================================================")

    # Open logger serial port
    try:
        ser_logger = serial.Serial(port, args.baud, timeout=1.0)
    except Exception as exc:
        print(f"[ERROR] Could not open serial port {port}: {exc}")
        sys.exit(1)

    # Optionally open DUT serial port
    ser_dut = None
    dut_log_file = None
    if args.dut_port:
        try:
            ser_dut = serial.Serial(args.dut_port, 115200, timeout=0.1)
            dut_log_file = open(dut_log_path, "w", encoding="utf-8")
            print(f"[+] Connected to DUT console on {args.dut_port}")
        except Exception as exc:
            print(f"[WARN] Could not connect to DUT port {args.dut_port}: {exc}")

    # Send START command to logging host
    time.sleep(1.0)
    ser_logger.reset_input_buffer()
    ser_logger.write(b"START\n")
    ser_logger.flush()

    csv_file = open(csv_file_path, "w", encoding="utf-8", buffering=1) # line-buffered
    header_written = False

    sample_count = 0
    rounds_completed = 0
    prev_comm_state = 0
    prev_train_state = 0
    start_wall_time = time.time()
    last_flush_time = start_wall_time
    last_activity_time = start_wall_time

    print(f"[+] Logging started. Waiting for phase triggers...")

    server_finished = threading.Event()

    def server_monitor():
        while not server_finished.is_set():
            try:
                import urllib.request
                req = urllib.request.Request("http://127.0.0.1:8080/status", headers={"User-Agent": "Logger"})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        s_data = json.loads(resp.read().decode("utf-8"))
                        if s_data.get("is_finished", False):
                            print("\n[+] FL Server signaled all global rounds are finished!")
                            server_finished.set()
                            break
            except Exception:
                # If server closed after substantial logging (>15s and >2000 samples), run is finished
                if sample_count > 2000 and (time.time() - start_wall_time) > 15.0:
                    print("\n[+] FL Server port closed. Benchmark completed!")
                    server_finished.set()
                    break
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=server_monitor, daemon=True)
    monitor_thread.start()

    try:
        train_seen_in_round = False

        while True:
            # Check duration limit
            elapsed = time.time() - start_wall_time
            if args.duration and elapsed >= args.duration:
                print(f"\n[*] Reached maximum specified duration ({args.duration} s). Stopping...")
                break

            if server_finished.is_set():
                # Capture an extra 1.5 seconds to record the final settling baseline
                time.sleep(1.5)
                print("[*] Benchmark run complete. Finalizing log...")
                break

            # Read DUT serial if enabled
            if ser_dut and ser_dut.in_waiting:
                dut_line = ser_dut.readline().decode("utf-8", errors="ignore")
                if dut_line:
                    dut_log_file.write(dut_line)
                    if any(k in dut_line for k in ["ROUND_COMPLETE", "Round completed", "FL_DONE"]):
                        print(f"  [DUT EVENT] {dut_line.strip()}")

            # Read from logger
            if ser_logger.in_waiting:
                raw_line = ser_logger.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                if raw_line.startswith("#"):
                    print(f"  [LOGGER] {raw_line}")
                    continue

                parts = raw_line.split(",")
                if len(parts) >= 6:
                    if not header_written:
                        if len(parts) >= 9:
                            csv_file.write(CSV_HEADER_9 + "\n")
                        else:
                            csv_file.write(CSV_HEADER_6 + "\n")
                        header_written = True

                    try:
                        t_ms = float(parts[0])
                        v_v = float(parts[1])
                        i_a = float(parts[2])
                        p_mw = float(parts[3])
                        g_train = int(parts[4])
                        g_comm = int(parts[5])

                        # Track true completed client iterations (TRAIN -> COMM Upload falling edge)
                        if g_train == 1 or g_comm == 1:
                            last_activity_time = time.time()

                        if g_train == 1:
                            train_seen_in_round = True
                        if train_seen_in_round and prev_comm_state == 1 and g_comm == 0:
                            rounds_completed += 1
                            train_seen_in_round = False
                            print(f"[+] Client Update {rounds_completed} completed (at t = {t_ms/1000.0:.2f}s)")

                        # Auto-conclude if idle settling occurs after sufficient client updates
                        if rounds_completed >= 50 and (time.time() - last_activity_time) > 4.0:
                            print("\n[+] Inactivity detected after 50+ client updates. Benchmark run complete!")
                            server_finished.set()

                        prev_train_state = g_train
                        prev_comm_state = g_comm

                        csv_file.write(raw_line + "\n")
                        sample_count += 1

                        if time.time() - last_flush_time > 1.0:
                            csv_file.flush()
                            last_flush_time = time.time()

                    except ValueError:
                        continue
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[!] User interrupted capture (Ctrl+C). Flushing and closing...")

    finally:
        # Send STOP command to logging host
        try:
            ser_logger.write(b"STOP\n")
            ser_logger.flush()
        except Exception:
            pass

        csv_file.flush()
        csv_file.close()
        ser_logger.close()

        if dut_log_file:
            dut_log_file.flush()
            dut_log_file.close()
        if ser_dut:
            ser_dut.close()

        total_elapsed = time.time() - start_wall_time
        effective_rate = sample_count / total_elapsed if total_elapsed > 0 else 0

        # Save run metadata
        metadata = {
            "epochs": args.epochs,
            "repetition": args.repetition,
            "target_rounds": args.rounds,
            "completed_rounds": rounds_completed,
            "total_samples": sample_count,
            "duration_sec": total_elapsed,
            "effective_sample_rate_hz": round(effective_rate, 2),
            "csv_path": str(csv_file_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("===============================================================")
        print(f"[SUCCESS] Power trace saved: {csv_file_path}")
        print(f"[*] Total Samples:        {sample_count:,}")
        print(f"[*] Total Duration:       {total_elapsed:.2f} s")
        print(f"[*] Effective Rate:       {effective_rate:.2f} Hz")
        print(f"[*] Rounds Recorded:      {rounds_completed}")
        print(f"[*] Metadata JSON:        {meta_path}")
        print("===============================================================")


if __name__ == "__main__":
    main()
