"""Draw the charts in results/ from the CSVs that bench.py writes."""
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BASELINE = "#2a78d6", "#a3a29c"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def read(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def short(name):
    return name.split(" ")[0] if name.startswith(("cub", "cublas")) else name


def hbar(ax, labels, values, colors, unit, ref=None, ref_label=None):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    top = max(values + ([ref] if ref else []))
    for yi, v in zip(y, values):
        ax.text(v + top * 0.01, yi, f"{v:,.0f} {unit}", va="center", fontsize=9, color=INK2, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    if ref:
        ax.axvline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(ref * 0.99, len(labels) - 1, ref_label, ha="right", va="center", fontsize=9,
                color=INK2)
    ax.set_xlim(0, top * 1.18)


def main():
    info = json.loads((RES / "system.json").read_text())

    # 1. Reduction bandwidth per kernel.
    rows = read("reduction.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [short(r["kernel"]) for r in rows]
    vals = [float(r["bandwidth_gbps"]) for r in rows]
    colors = [BASELINE if r["kernel"].startswith("cub") else BLUE for r in rows]
    hbar(ax, labels, vals, colors, "GB/s", info["peak_dram_gbps"],
         f"DRAM peak {info['peak_dram_gbps']:.0f} GB/s")
    ax.set_xlabel("Effective bandwidth (GB/s), higher is better")
    ax.set_title("Parallel sum of 32M floats (128 MiB)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "reduction_bandwidth.png", dpi=160)

    # 2. SGEMM throughput per kernel at the largest size.
    rows = read("sgemm.csv")
    big = max(int(r["size"]) for r in rows)
    at_big = [r for r in rows if int(r["size"]) == big]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [short(r["kernel"]) for r in at_big]
    vals = [float(r["gflops"]) for r in at_big]
    colors = [BASELINE if r["kernel"].startswith("cublas") else BLUE for r in at_big]
    hbar(ax, labels, vals, colors, "GFLOP/s")
    ax.set_xlabel("GFLOP/s, higher is better")
    ax.set_title(f"FP32 SGEMM, {big}x{big} matrices", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "sgemm_kernels.png", dpi=160)

    # 3. SGEMM throughput vs matrix size, one line per kernel.
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for color, k in zip(SERIES, kernels):
        pts = [(int(r["size"]), float(r["gflops"])) for r in rows if r["kernel"] == k]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=short(k))
    ax.set_xscale("log", base=2)
    sizes = sorted({int(r["size"]) for r in rows})
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Matrix size N (N x N x N)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("SGEMM throughput vs matrix size", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "sgemm_scaling.png", dpi=160)
    print("Wrote reduction_bandwidth.png, sgemm_kernels.png, sgemm_scaling.png")


if __name__ == "__main__":
    main()
