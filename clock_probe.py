"""Measure the SM clock and power the GPU actually sustains during a long FP32 SGEMM.

Laptop GPUs are power-capped, so the sustained clock can sit well below the boost clock
used for the theoretical peak. Writes results/sustained_clock.json.
"""
import json
import os
import pathlib
import subprocess
import threading
import time

os.environ.setdefault("CUPY_TF32", "0")
import cupy as cp

ROOT = pathlib.Path(__file__).resolve().parent
samples, stop = [], threading.Event()


def sample():
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        clk, pwr, temp = (float(x) for x in out.stdout.strip().split(","))
        samples.append((clk, pwr, temp))
        time.sleep(0.25)


n = 4096
A = cp.random.rand(n, n, dtype=cp.float32)
B = cp.random.rand(n, n, dtype=cp.float32)
A @ B
cp.cuda.Device().synchronize()

t = threading.Thread(target=sample)
t.start()
start, iters = time.perf_counter(), 0
while time.perf_counter() - start < 10:
    for _ in range(10):
        A @ B
    cp.cuda.Device().synchronize()
    iters += 10
elapsed = time.perf_counter() - start
stop.set()
t.join()

steady = samples[len(samples) // 4:]      # drop ramp-up samples
p = cp.cuda.runtime.getDeviceProperties(0)
clk = sum(s[0] for s in steady) / len(steady)
res = {
    "sustained_sm_clock_mhz": round(clk),
    "avg_power_w": round(sum(s[1] for s in steady) / len(steady), 1),
    "max_temp_c": max(s[2] for s in steady),
    "cublas_gflops_10s": round(2 * n**3 * iters / elapsed / 1e9, 1),
    "peak_fp32_gflops_at_sustained_clock": round(p["multiProcessorCount"] * 128 * 2 * clk / 1e3, 1),
}
print(json.dumps(res, indent=2))
(ROOT / "results" / "sustained_clock.json").write_text(json.dumps(res, indent=2))
