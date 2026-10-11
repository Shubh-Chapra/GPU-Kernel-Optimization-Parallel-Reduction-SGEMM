# Profile every kernel with Nsight Compute. Writes:
#   results/ncu_metrics.csv   selected metrics, one row per kernel (summarised by ncu_summary.py)
#   results/ncu_full.ncu-repz full report, open it in the Nsight Compute GUI
# On Windows, GPU performance counters need admin rights (or "Allow access to all users"
# under NVIDIA Control Panel > Developer > Manage GPU Performance Counters).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$filter = "regex:^(reduce|sgemm)"
# ncu locks clocks to base by default. On this laptop that also drops the memory clock to
# ~10% of normal, which makes every kernel look DRAM-bound, so profile at normal clocks.
$clocks = @("--clock-control", "none")

$metrics = @(
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__thread_inst_executed_per_inst_executed.ratio"
) -join ","

& $ncu -k $filter @clocks --csv --page raw --metrics $metrics $py -W ignore profile_kernels.py 2>&1 |
    Out-File -Encoding utf8 results\ncu_metrics.csv
& $ncu -k $filter @clocks --set full -f -o results\ncu_full $py -W ignore profile_kernels.py 2>&1 |
    Out-File -Encoding utf8 results\ncu_full_log.txt
"done" | Out-File -Encoding utf8 results\ncu_done.txt
