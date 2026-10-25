#!/usr/bin/env python3
"""
server/aggregate.py

Federated Learning Aggregation Server (FedAvg) for Edge vs TinyML KWS benchmark:
- Supports both single-board sequential simulation of K virtual clients (realistic
  for solo-author testbed) and concurrent multi-board execution.
- Endpoints:
    GET  /config            -> Global training parameters and current round state.
    GET  /model             -> Current global head weights (binary or JSON).
    GET  /partition/<k>     -> Local dataset partition metadata/features for client k.
    POST /update            -> Upload local head weights from client (binary or JSON).
    GET  /status            -> Monitor current aggregation progress.
- Computes FedAvg across K client updates:
    W_global = sum(n_k / N * W_k), b_global = sum(n_k / N * b_k)
- Logs per-round and per-client timing and byte transfer to:
    data/processed/fl_rounds_log.csv
- Automatically terminates cleanly after R total rounds.
"""

import os
import sys
import time
import json
import struct
import socket
import argparse
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INITIAL_HEAD = REPO_ROOT / "data" / "head_weights.npz"
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "processed" / "fl_rounds_log.csv"

NUM_INPUTS = 64
NUM_CLASSES = 12
HEAD_WEIGHTS_COUNT = NUM_INPUTS * NUM_CLASSES # 768
HEAD_BIAS_COUNT = NUM_CLASSES                  # 12
HEAD_TOTAL_PARAMS = HEAD_WEIGHTS_COUNT + HEAD_BIAS_COUNT # 780
EXPECTED_BINARY_SIZE = HEAD_TOTAL_PARAMS * 4    # 3,120 bytes


class FederatedState:
    """Thread-safe state manager for FedAvg rounds and client updates."""
    def __init__(self, num_clients: int, total_rounds: int, epochs: int,
                 initial_weights: np.ndarray, initial_biases: np.ndarray,
                 log_path: Path, mode: str = "sequential", settle_delay: int = 60):
        self.lock = threading.Lock()
        self.K = num_clients
        self.total_rounds = total_rounds
        self.E = epochs
        self.settle_delay = settle_delay
        self.mode = mode
        self.log_path = log_path
        self.current_round = 1
        self.current_client_turn = 0 # Used in sequential single-board mode
        self.is_finished = False

        # Global model parameters
        self.global_weights = initial_weights.copy().astype(np.float32)
        self.global_biases = initial_biases.copy().astype(np.float32)

        # Buffer for current round client updates: client_id -> (weights, biases, num_samples, bytes_transferred, start_time, duration)
        self.round_updates = {}
        self.client_start_times = {}
        self.log_buffer = []

        # Initialize CSV log file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp_iso,global_round,client_id,epochs,update_size_bytes,duration_sec,num_samples,round_completed\n")

    def flush_log(self):
        """Flushes buffered log entries to disk."""
        if not self.log_buffer:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
        except Exception as exc:
            print(f"[WARN] Failed to flush log to {self.log_path}: {exc}")

    def get_client_assignment(self):
        """Returns the client ID assigned for the next training pass."""
        with self.lock:
            if self.mode == "sequential":
                return self.current_client_turn
            return None

    def record_client_start(self, client_id: int):
        with self.lock:
            self.client_start_times[client_id] = time.time()

    def submit_update(self, client_id: int, weights: np.ndarray, biases: np.ndarray,
                      num_samples: int, update_bytes: int):
        with self.lock:
            now = time.time()
            start_t = self.client_start_times.get(client_id, now)
            duration = now - start_t

            self.round_updates[client_id] = {
                "weights": weights.copy(),
                "biases": biases.copy(),
                "samples": num_samples,
                "bytes": update_bytes,
                "duration": duration,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

            # Buffer client log line in memory (no disk I/O during request!)
            self.log_buffer.append(
                f"{self.round_updates[client_id]['timestamp']},{self.current_round},{client_id},"
                f"{self.E},{update_bytes},{duration:.3f},{num_samples},0\n"
            )

            print(f"[+] Received update from Client {client_id} (Round {self.current_round}/{self.total_rounds}): "
                  f"{update_bytes:,} bytes, {num_samples} samples, {duration:.2f}s")

            # Advance sequential turn
            if self.mode == "sequential":
                self.current_client_turn = (self.current_client_turn + 1) % self.K

            # Check if all K clients have submitted for this round
            if len(self.round_updates) >= self.K:
                self.aggregate_round()

    def aggregate_round(self):
        """Executes FedAvg aggregation across the K client updates."""
        print(f"\n[*] --- Aggregating Round {self.current_round}/{self.total_rounds} across {len(self.round_updates)} clients ---")
        total_samples = sum(u["samples"] for u in self.round_updates.values())
        if total_samples <= 0:
            total_samples = len(self.round_updates)

        new_weights = np.zeros_like(self.global_weights)
        new_biases = np.zeros_like(self.global_biases)

        total_bytes = 0
        for cid, u in self.round_updates.items():
            weight_factor = u["samples"] / total_samples
            new_weights += weight_factor * u["weights"]
            new_biases += weight_factor * u["biases"]
            total_bytes += u["bytes"]
            print(f"    - Client {cid}: weight_factor={weight_factor:.4f} ({u['samples']} samples)")

        self.global_weights = new_weights
        self.global_biases = new_biases

        # Buffer round summary log in memory
        round_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.log_buffer.append(
            f"{round_ts},{self.current_round},ALL,{self.E},{total_bytes},0.000,{total_samples},1\n"
        )
        self.flush_log()

        print(f"[SUCCESS] Round {self.current_round} FedAvg aggregation complete. New global head updated.")

        # Check for experiment completion
        if self.current_round >= self.total_rounds:
            self.is_finished = True
            print(f"\n===============================================================")
            print(f"  [FINISHED] All {self.total_rounds} global FL rounds completed cleanly!")
            print(f"  Log file saved to: {self.log_path}")
            print(f"===============================================================\n")
        else:
            self.current_round += 1
            self.round_updates.clear()
            self.client_start_times.clear()


class FLRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for FL clients."""

    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.request.settimeout(15.0)

    def handle(self):
        self.close_connection = True
        super().handle()

    def _send_response_data(self, data: bytes, status=200, content_type="application/json"):
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except (socket.timeout, ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        state: FederatedState = self.server.state

        if self.path == "/config":
            assigned_client = state.get_client_assignment()
            state.record_client_start(assigned_client)

            response = {
                "round": state.current_round,
                "total_rounds": state.total_rounds,
                "epochs": state.E,
                "num_clients": state.K,
                "assigned_client_id": assigned_client,
                "mode": state.mode,
                "is_finished": state.is_finished,
                "settle_delay_ms": state.settle_delay,
                "input_dim": NUM_INPUTS,
                "output_classes": NUM_CLASSES,
                "expected_bytes": EXPECTED_BINARY_SIZE,
            }
            self._send_response_data(json.dumps(response).encode("utf-8"), 200, "application/json")

        elif self.path == "/model" or self.path == "/model/bin":
            # Return current global head as flat float32 binary
            bin_data = state.global_weights.tobytes() + state.global_biases.tobytes()
            self._send_response_data(bin_data, 200, "application/octet-stream")

        elif self.path == "/model/json":
            response = {
                "round": state.current_round,
                "weights": state.global_weights.tolist(),
                "biases": state.global_biases.tolist(),
            }
            self._send_response_data(json.dumps(response).encode("utf-8"), 200, "application/json")

        elif self.path.startswith("/partition/"):
            try:
                cid = int(self.path.split("/")[-1])
                response = {
                    "client_id": cid,
                    "num_samples": 100,
                    "seed": 42 + cid,
                    "description": f"Virtual client partition {cid}/{state.K}"
                }
                self._send_response_data(json.dumps(response).encode("utf-8"), 200, "application/json")
            except ValueError:
                self._send_response_data(b'{"error": "Invalid client_id"}', 400, "application/json")

        elif self.path == "/status":
            response = {
                "current_round": state.current_round,
                "total_rounds": state.total_rounds,
                "epochs": state.E,
                "clients_submitted": list(state.round_updates.keys()),
                "total_clients": state.K,
                "is_finished": state.is_finished,
            }
            self._send_response_data(json.dumps(response).encode("utf-8"), 200, "application/json")

        else:
            self._send_response_data(b'{"error": "Endpoint not found"}', 404, "application/json")

    def do_POST(self):
        state: FederatedState = self.server.state

        if self.path == "/update" or self.path == "/update/bin":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                cid_header = self.headers.get("X-Client-ID")
                client_id = int(cid_header) if (cid_header is not None and cid_header.strip().isdigit()) else state.get_client_assignment()
                num_samples = int(self.headers.get("X-Num-Samples", 100))

                body = self.rfile.read(content_len)

                if len(body) == EXPECTED_BINARY_SIZE:
                    weights_bytes = body[:HEAD_WEIGHTS_COUNT * 4]
                    bias_bytes = body[HEAD_WEIGHTS_COUNT * 4:]
                    weights = np.frombuffer(weights_bytes, dtype=np.float32).reshape((NUM_INPUTS, NUM_CLASSES))
                    biases = np.frombuffer(bias_bytes, dtype=np.float32)

                    state.submit_update(client_id, weights, biases, num_samples, len(body))
                    resp = json.dumps({"status": "accepted", "round": state.current_round}).encode("utf-8")
                    self._send_response_data(resp, 200, "application/json")
                else:
                    print(f"[WARN] Received unexpected payload size: {len(body)} (expected {EXPECTED_BINARY_SIZE})")
                    try:
                        data = json.loads(body.decode("utf-8"))
                        weights = np.array(data["weights"], dtype=np.float32).reshape((NUM_INPUTS, NUM_CLASSES))
                        biases = np.array(data["biases"], dtype=np.float32)
                        cid = data.get("client_id", client_id)
                        samples = data.get("samples", num_samples)

                        state.submit_update(cid, weights, biases, samples, len(body))
                        resp = json.dumps({"status": "accepted", "round": state.current_round}).encode("utf-8")
                        self._send_response_data(resp, 200, "application/json")
                    except Exception as exc:
                        err = json.dumps({"error": f"Invalid payload size {len(body)}: {exc}"}).encode("utf-8")
                        self._send_response_data(err, 400, "application/json")
            except Exception as exc:
                print(f"[WARN] Connection error during POST: {exc}")
        else:
            self._send_response_data(b'{"error": "Endpoint not found"}', 404, "application/json")

    def log_message(self, format, *args):
        # Suppress routine GET logging to keep console clean
        if "GET /status" in args[0] or "GET /config" in args[0]:
            return
        super().log_message(format, *args)


def load_initial_weights(npz_path: Path):
    """Loads baseline head weights from NumPy archive or generates random baseline."""
    if npz_path.exists():
        print(f"[*] Loading initial head weights from: {npz_path}")
        data = np.load(npz_path)
        w = data["weights"].astype(np.float32)
        b = data["biases"].astype(np.float32)
        return w, b
    else:
        print(f"[WARN] Initial weights not found at {npz_path}. Initializing with zeros.")
        w = np.zeros((NUM_INPUTS, NUM_CLASSES), dtype=np.float32)
        b = np.zeros((NUM_CLASSES,), dtype=np.float32)
        return w, b


def main():
    parser = argparse.ArgumentParser(description="FedAvg Aggregation Server for Edge vs TinyML benchmark.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind HTTP server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to listen for client connections (default: 8080).")
    parser.add_argument("-K", "--num-clients", type=int, default=5,
                        help="Number of simulated/physical clients per FL round (default: 5).")
    parser.add_argument("-R", "--rounds", type=int, default=10,
                        help="Total number of global FedAvg rounds to execute (default: 10).")
    parser.add_argument("-E", "--epochs", type=int, default=5,
                        help="Local epochs E per round (default: 5).")
    parser.add_argument("--settle-delay", type=int, default=60,
                        help="Inter-phase settling delay in milliseconds (default: 60).")
    parser.add_argument("--mode", type=str, choices=["sequential", "concurrent"], default="sequential",
                        help="Client multiplexing mode: 'sequential' (single board standing in for K clients) "
                              "or 'concurrent' (K distinct physical boards).")
    parser.add_argument("--initial-weights", type=Path, default=DEFAULT_INITIAL_HEAD,
                        help="Path to initial head_weights.npz")
    parser.add_argument("--output-log", type=Path, default=DEFAULT_LOG_PATH,
                        help="Path to save fl_rounds_log.csv")
    args = parser.parse_args()

    print("===============================================================")
    print("      Federated Learning Aggregation Server (FedAvg)          ")
    print("===============================================================")
    print(f"[*] Binding:          http://{args.host}:{args.port}")
    print(f"[*] Virtual Clients:  K = {args.num_clients} ({args.mode} mode)")
    print(f"[*] Global Rounds:    R = {args.rounds}")
    print(f"[*] Local Epochs (E): E = {args.epochs}")
    print(f"[*] Settle Delay:     {args.settle_delay} ms")
    print(f"[*] Model Head Size:  {EXPECTED_BINARY_SIZE:,} bytes ({HEAD_TOTAL_PARAMS} float32 params)")
    print(f"[*] Log Destination:  {args.output_log}")
    print("===============================================================\n")

    initial_w, initial_b = load_initial_weights(args.initial_weights)

    state = FederatedState(
        num_clients=args.num_clients,
        total_rounds=args.rounds,
        epochs=args.epochs,
        settle_delay=args.settle_delay,
        initial_weights=initial_w,
        initial_biases=initial_b,
        log_path=args.output_log,
        mode=args.mode
    )

    server = ThreadingHTTPServer((args.host, args.port), FLRequestHandler)
    server.daemon_threads = True
    server.timeout = 0.5
    server.state = state

    print(f"[+] Server listening on port {args.port}. Waiting for ESP32-S3 clients...")

    try:
        while not state.is_finished:
            server.handle_request()
        # Keep serving briefly so logger & DUT receive final is_finished status
        t_drain = time.time()
        while time.time() - t_drain < 3.5:
            server.handle_request()
    except KeyboardInterrupt:
        print("\n[!] Server interrupted by user (Ctrl+C). Cleaning up...")
    finally:
        state.flush_log()
        server.server_close()
        print(f"[+] Server stopped cleanly. Final round log at: {args.output_log}")


if __name__ == "__main__":
    main()
