"""Turn results/ncu_metrics.csv (raw ncu --csv output) into results/ncu_summary.csv."""
import csv
import pathlib

RES = pathlib.Path(__file__).resolve().parent / "results"

# (column name in the summary, ncu metric)
COLUMNS = [
    ("time", "gpu__time_duration.sum"),
    ("sm_clock", "sm__cycles_elapsed.avg.per_second"),
    ("dram_clock", "dram__cycles_elapsed.avg.per_second"),
    ("dram_pct","dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("fma_pipe_pct", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("achieved_occ_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("theoretical_occ_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("regs", "launch__registers_per_thread"),
    ("block_limit_regs", "launch__occupancy_limit_registers"),
    ("smem_ld_conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
    ("smem_st_conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
    ("active_threads_per_warp", "smsp__thread_inst_executed_per_inst_executed.ratio"),
]
SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
TIME_SCALE = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3,                       # to ms
              "nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3}


def num(s):
    return float(s.replace(",", "")) if s not in ("", "n/a") else float("nan")


def main():
    lines = (RES / "ncu_metrics.csv").read_text(encoding="utf-8-sig").splitlines()
    rows = list(csv.reader(l for l in lines if l.startswith('"')))
    header, units, data = rows[0], rows[1], rows[2:]
    col = {name: i for i, name in enumerate(header)}

    out = []
    for r in data:
        row = {"kernel": r[col["Kernel Name"]], "grid": r[col["Grid Size"]],
               "block": r[col["Block Size"]]}
        for name, metric in COLUMNS:
            v = num(r[col[metric]])
            if name == "time":
                v *= TIME_SCALE[units[col[metric]]]
                name = "time_ms"
            elif name.endswith("clock"):
                name = f"{name}_{units[col[metric]].replace('/', '_per_')}"
            row[name] = round(v, 3) if name == "time_ms" else round(v, 1)
        # 32 threads reading 32 consecutive floats = 4 sectors of 32 B; float4 loads = 16.
        row["sectors_per_load_request"] = round(num(r[col[SECTORS]]) / num(r[col[REQUESTS]]), 1)
        out.append(row)

    with open(RES / "ncu_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
