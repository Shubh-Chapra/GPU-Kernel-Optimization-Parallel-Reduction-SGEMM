"""Benchmark harness for the CUDA kernels in kernels/.

Compiles kernels/*.cuh at runtime with NVRTC (through CuPy), checks every kernel against a
reference, times it with CUDA events, and writes results/*.csv and results/system.json.

    python bench.py              # full run
    python bench.py --quick      # smaller sizes, for a fast sanity check
"""
import argparse
import csv
import json
import os
import pathlib

os.environ.setdefault("CUPY_TF32", "0")   # keep the cuBLAS baseline in true FP32

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"

REDUCTION_KERNELS = [
    # name, elements summed per thread before the shared-memory tree
    ("reduce0_interleaved_divergent", 1),
    ("reduce1_interleaved_bankconflict", 1),
    ("reduce2_sequential", 1),
    ("reduce3_first_add", 2),
    ("reduce4_warp_shuffle", 2),
]
RED_BLOCK = 256

SGEMM_KERNELS = ["sgemm0_naive", "sgemm1_coalesced", "sgemm2_smem_tiled",
                 "sgemm3_1d_blocktile", "sgemm4_2d_blocktile", "sgemm5_vectorized"]


def load_module(filename):
    src = (ROOT / "kernels" / filename).read_text()
    return cp.RawModule(code=src, options=("--std=c++17",))


def gpu_ms(fn, n_repeat, n_warmup=3):
    """Median GPU time of fn() in milliseconds, measured with CUDA events."""
    res = benchmark(fn, (), n_repeat=n_repeat, n_warmup=n_warmup)
    return float(np.median(res.gpu_times[0])) * 1e3


def system_info():
    p = cp.cuda.runtime.getDeviceProperties(0)
    sms = p["multiProcessorCount"]
    cores_per_sm = 128 if (p["major"], p["minor"]) in [(8, 6), (8, 7), (8, 9)] else 64
    clock_ghz = p["clockRate"] / 1e6
    peak_gflops = sms * cores_per_sm * 2 * clock_ghz              # FMA = 2 FLOPs
    peak_bw = 2 * p["memoryClockRate"] * 1e3 * (p["memoryBusWidth"] / 8) / 1e9   # DDR
    return {
        "gpu": p["name"].decode(),
        "compute_capability": f"{p['major']}.{p['minor']}",
        "sms": sms,
        "fp32_cores": sms * cores_per_sm,
        "boost_clock_ghz": clock_ghz,
        "peak_fp32_gflops": round(peak_gflops, 1),
        "peak_dram_gbps": round(peak_bw, 1),
        "l2_mb": p["l2CacheSize"] / 2**20,
        "max_threads_per_sm": p["maxThreadsPerMultiProcessor"],
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "driver": cp.cuda.runtime.driverGetVersion(),
        "cupy": cp.__version__,
    }


def kernel_resources(fn):
    return {"regs_per_thread": fn.num_regs, "static_smem_bytes": fn.shared_size_bytes}


# ----------------------------------------------------------------------------- reduction

def bench_reduction(info, n, n_repeat):
    mod = load_module("reduction.cuh")
    rng = np.random.default_rng(0)
    h = rng.random(n, dtype=np.float32)
    ref = float(h.astype(np.float64).sum())
    d = cp.asarray(h)
    bytes_read = n * 4
    rows = []

    def record(name, ms, value, extra=None):
        rel_err = abs(value - ref) / abs(ref)
        ok = rel_err < 1e-4
        gbps = bytes_read / (ms * 1e-3) / 1e9
        row = {"kernel": name, "n": n, "time_ms": round(ms, 4), "bandwidth_gbps": round(gbps, 1),
               "pct_peak_bw": round(100 * gbps / info["peak_dram_gbps"], 1),
               "rel_error": f"{rel_err:.2e}", "correct": ok}
        row.update(extra or {})
        rows.append(row)
        print(f"  {name:34s} {ms:8.3f} ms  {gbps:7.1f} GB/s  ({row['pct_peak_bw']:5.1f}% peak)"
              f"  err {rel_err:.1e}  {'OK' if ok else 'WRONG'}")

    # Multi-pass kernels: relaunch on the partial sums until one value remains.
    for name, per_thread in REDUCTION_KERNELS:
        fn = mod.get_function(name)
        sizes, bufs, m = [], [], n
        while m > 1:
            blocks = (m + RED_BLOCK * per_thread - 1) // (RED_BLOCK * per_thread)
            sizes.append((m, blocks))
            bufs.append(cp.empty(blocks, dtype=cp.float32))
            m = blocks

        def run(fn=fn, sizes=sizes, bufs=bufs):
            src = d
            for (m, blocks), out in zip(sizes, bufs):
                fn((blocks,), (RED_BLOCK,), (src, out, np.uint32(m)))
                src = out
            return src

        result = float(run()[0])
        record(name, gpu_ms(run, n_repeat), result,
               {"launches": len(sizes), **kernel_resources(fn)})

    # Single-pass kernel: grid sized to fill every SM, plus a sweep over grid size.
    fn5 = mod.get_function("reduce5_gridstride_vec4")
    out = cp.zeros(1, dtype=cp.float32)
    full = info["max_threads_per_sm"] // RED_BLOCK          # resident blocks per SM

    def run5(blocks):
        out.fill(0)
        fn5((blocks,), (RED_BLOCK,), (d, out, np.uint32(n)))
        return out

    sweep = []
    for per_sm in [1, 2, 4, full, 2 * full, 4 * full, 8 * full, 16 * full]:
        blocks = per_sm * info["sms"]
        ms = gpu_ms(lambda: run5(blocks), n_repeat)
        sweep.append({"blocks_per_sm": per_sm, "blocks": blocks, "time_ms": round(ms, 4),
                      "bandwidth_gbps": round(bytes_read / (ms * 1e-3) / 1e9, 1)})
    best = min(sweep, key=lambda r: r["time_ms"])
    result = float(run5(best["blocks"])[0])
    record("reduce5_gridstride_vec4", best["time_ms"], result,
           {"launches": 1, "blocks": best["blocks"], **kernel_resources(fn5)})

    # Library baseline: cupy.sum dispatches to CUB's device reduction.
    result = float(cp.sum(d))
    record("cub (cupy.sum)", gpu_ms(lambda: cp.sum(d), n_repeat), result)
    return rows, sweep


# ----------------------------------------------------------------------------- sgemm

def sgemm_launch(name, M, N, K):
    """Grid and block shape for each SGEMM kernel."""
    if name == "sgemm0_naive":
        return ((M + 31) // 32, (N + 31) // 32), (32, 32)
    if name in ("sgemm1_coalesced", "sgemm2_smem_tiled"):
        return ((N + 31) // 32, (M + 31) // 32), (32, 32)
    if name == "sgemm3_1d_blocktile":
        assert M % 64 == 0 and N % 64 == 0 and K % 8 == 0
        return (N // 64, M // 64), (512,)
    assert M % 128 == 0 and N % 128 == 0 and K % 8 == 0
    return (N // 128, M // 128), (256,)


def bench_sgemm(info, sizes, n_repeat):
    mod = load_module("sgemm.cuh")
    rows = []
    for s in sizes:
        M = N = K = s
        rng = np.random.default_rng(s)
        A = cp.asarray(rng.uniform(-1, 1, (M, K)).astype(np.float32))
        B = cp.asarray(rng.uniform(-1, 1, (K, N)).astype(np.float32))
        C = cp.zeros((M, N), dtype=cp.float32)
        flops = 2.0 * M * N * K
        ref = A @ B                                           # cuBLAS SGEMM
        scale = float(cp.abs(ref).max())
        # Slow kernels at big sizes take seconds per call; fewer repeats keep runs short.
        reps = max(3, min(n_repeat, int(n_repeat * (1024 / s) ** 3)))

        def record(name, ms, out, extra=None):
            err = float(cp.abs(out - ref).max()) / scale
            ok = err < 1e-4
            gflops = flops / (ms * 1e-3) / 1e9
            row = {"kernel": name, "size": s, "time_ms": round(ms, 4),
                   "gflops": round(gflops, 1),
                   "pct_peak": round(100 * gflops / info["peak_fp32_gflops"], 1),
                   "max_rel_error": f"{err:.2e}", "correct": ok}
            row.update(extra or {})
            rows.append(row)
            print(f"  {s:5d}  {name:22s} {ms:9.3f} ms  {gflops:8.1f} GFLOP/s"
                  f"  ({row['pct_peak']:5.1f}% peak)  err {err:.1e}  {'OK' if ok else 'WRONG'}")

        for name in SGEMM_KERNELS:
            fn = mod.get_function(name)
            grid, block = sgemm_launch(name, M, N, K)
            args = (np.int32(M), np.int32(N), np.int32(K), np.float32(1.0), A, B,
                    np.float32(0.0), C)

            def run(fn=fn, grid=grid, block=block, args=args):
                fn(grid, block, args)

            C.fill(0)
            run()
            out = C.copy()
            record(name, gpu_ms(run, reps, n_warmup=1), out, kernel_resources(fn))

        record("cublas (cupy.matmul)", gpu_ms(lambda: A @ B, n_repeat), A @ B)
        del A, B, C, ref
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, few repeats")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    info = system_info()
    print(json.dumps(info, indent=2))

    n = 1 << (20 if args.quick else 25)
    sizes = [128, 256] if args.quick else [256, 512, 1024, 2048, 4096]
    reps = 5 if args.quick else 50

    print(f"\nReduction: sum of {n:,} floats ({n * 4 / 2**20:.0f} MiB)")
    red_rows, sweep = bench_reduction(info, n, reps)
    print("\nSGEMM: C = A @ B, square FP32 matrices")
    gemm_rows = bench_sgemm(info, sizes, reps)

    if not args.quick:
        (RESULTS / "system.json").write_text(json.dumps(info, indent=2))
        write_csv(RESULTS / "reduction.csv", red_rows)
        write_csv(RESULTS / "reduction_grid_sweep.csv", sweep)
        write_csv(RESULTS / "sgemm.csv", gemm_rows)
        print(f"\nWrote results to {RESULTS}")

    bad = [r["kernel"] for r in red_rows + gemm_rows if not r["correct"]]
    if bad:
        raise SystemExit(f"FAILED correctness check: {bad}")


if __name__ == "__main__":
    main()
