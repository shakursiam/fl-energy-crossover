"""Campaign 4 host runner: record the INA238 logger stream during the DUT sleep cycles.

Board A (Board_A_Sleep.ino) runs autonomously; this script only listens.
  - Logger (Board B) CSV stream  -> <out>/power_trace.csv
  - DUT (Board A) serial output  -> <out>/dut.log   (best effort; the port disappears in deep
    sleep, and it is opened with DTR/RTS low so that opening it never resets the DUT)
The run ends after the second 500 ms session marker (code 7) or after --timeout-min.

Usage: python runner/run_sleep_campaign.py --logger COM18 --dut COM7 --out data/raw_sleep/session_01
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime

import serial

HEADER = 'timestamp_ms,voltage_V,current_A,power_mW,gpio_train,gpio_comm,raw_phase_code,debounced_phase_code,dropped_samples'


def dut_reader(port, path, stop, delay_s):
    stop.wait(delay_s)  # leave the port free while the DUT is being flashed
    with open(path, 'a', encoding='utf-8', errors='replace') as f:
        while not stop.is_set():
            try:
                s = serial.Serial()
                s.port, s.baudrate, s.timeout = port, 115200, 0.3
                s.dtr = False
                s.rts = False
                s.open()
                f.write(f'## {time.time():.3f} port opened\n'); f.flush()
                while not stop.is_set():
                    data = s.read(4096)
                    if data:
                        for line in data.decode(errors='replace').splitlines(True):
                            f.write(f'{time.time():.3f} {line}' if line.strip() else line)
                        f.flush()
            except (serial.SerialException, OSError):
                time.sleep(0.5)
            finally:
                try:
                    s.close()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logger', required=True)
    ap.add_argument('--dut', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dut-delay-s', type=float, default=45)
    ap.add_argument('--timeout-min', type=float, default=70)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    stop = threading.Event()
    if a.dut:
        th = threading.Thread(target=dut_reader, args=(a.dut, os.path.join(a.out, 'dut.log'), stop, a.dut_delay_s),
                              daemon=True)
        th.start()

    L = serial.Serial(a.logger, 921600, timeout=0.2)
    t_start = time.time()
    n_rows, run7, session_marks, header_lines = 0, 0, 0, []
    buf = b''
    with open(os.path.join(a.out, 'power_trace.csv'), 'w', encoding='utf-8') as f:
        f.write(HEADER + '\n')
        while True:
            buf += L.read(65536)
            *lines, buf = buf.split(b'\n')
            for raw in lines:
                line = raw.decode(errors='replace').strip()
                if not line:
                    continue
                if line.startswith('#'):
                    header_lines.append(line)
                    continue
                parts = line.split(',')
                if len(parts) != 9 or not parts[0].isdigit():
                    continue
                f.write(line + '\n')
                n_rows += 1
                if parts[6] == '7':
                    run7 += 1
                else:
                    if run7 >= 400:
                        session_marks += 1
                        print(f'[{time.time() - t_start:7.1f} s] session marker {session_marks}', flush=True)
                    run7 = 0
            if n_rows and n_rows % 60000 < 200:
                print(f'[{time.time() - t_start:7.1f} s] {n_rows} samples', flush=True)
            if session_marks >= 2 or time.time() - t_start > a.timeout_min * 60:
                break
    L.close()
    time.sleep(2)
    stop.set()
    meta = {'started': datetime.fromtimestamp(t_start).isoformat(timespec='seconds'),
            'duration_s': round(time.time() - t_start, 1), 'samples': n_rows,
            'completed': session_marks >= 2, 'logger_port': a.logger, 'dut_port': a.dut,
            'logger_header': header_lines}
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != 'logger_header'}))


if __name__ == '__main__':
    main()
