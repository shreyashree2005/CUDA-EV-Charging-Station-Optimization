
"""
graphs.py
=========
Every figure here is plotted from a CSV that run_experiment.py / ablation.py
actually wrote — this file makes ZERO calls into core.py. That's a
deliberate change from an earlier version, where fig_dual_peak,
fig_hourly_curve, and fig_map imported core.py and recomputed values live
at plot time. Those numbers were already correct (same deterministic
functions), but "correct" isn't the same guarantee as "traceable to a
saved artifact a reviewer can open and check independently" — reading only
from CSVs that run_experiment.py wrote means every figure has an audit
trail, and a figure and its underlying data can never silently drift apart
from each other.

CHANGES IN THIS VERSION, and why:

1. fig_convergence's title/legend are now built FROM the CSV's own columns
   instead of being hardcoded text, so the title can never claim a curve
   that isn't actually plotted (or vice versa).
2. fig_pareto now uses a bare scatter plot — no connecting line. Points in
   a Pareto archive are discrete solutions; drawing a line through them
   visually implies a continuum of achievable trade-offs between them,
   which isn't true (nothing was evaluated in the gaps).
3. fig_dual_peak, fig_hourly_curve, fig_map now read table1b_dual_peak.csv,
   table1c_hourly_citywide.csv, table0_wards.csv respectively, instead of
   importing core.py.
4. fig_speedup / fig_speedup_vs_dataset_size now read the gpu_available
   column those CSVs carry and put an explicit "(CPU fallback — no GPU
   detected)" qualifier directly in the figure title when it's False, so
   the figure is honest about what it shows without requiring the reader
   to cross-reference Table II or console output.

Run backend/run_experiment.py and backend/ablation.py first to produce the
CSVs this script reads.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def rpath(name):
    return os.path.join(RESULTS_DIR, name)


def savefig(name):
    plt.tight_layout()
    plt.savefig(rpath(name), dpi=150)
    plt.close()
    print(f"  wrote {name}")


def fig_demand():
    df = pd.read_csv(rpath("table1_demand.csv"))
    plt.figure(figsize=(7, 4))
    plt.bar(df["ward"], df["demand_kwh"])
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.ylabel("Demand (kWh)")
    plt.title("Ward-level EV Charging Demand (temporal model, coincident peak hour)")
    savefig("fig1_demand.png")


def fig_dual_peak():
    df = pd.read_csv(rpath("table1b_dual_peak.csv"))
    order = (-df[["morning_kwh", "evening_kwh"]].max(axis=1)).argsort().values
    plt.figure(figsize=(8, 4))
    x = range(len(df))
    plt.bar([i - 0.2 for i in x], df["morning_kwh"].values[order], width=0.4,
            label="Morning component (commercial)", color="#e8ff47")
    plt.bar([i + 0.2 for i in x], df["evening_kwh"].values[order], width=0.4,
            label="Evening component (residential)", color="#4c9eff")
    plt.xticks(list(x), df["ward"].values[order], rotation=80, ha="right", fontsize=6)
    plt.ylabel("Demand (kWh)")
    plt.title("Morning vs Evening Demand Components at Coincident Peak Hour")
    plt.legend()
    savefig("fig1b_dual_peak_demand.png")


def fig_hourly_curve():
    df = pd.read_csv(rpath("table1c_hourly_citywide.csv"))
    peak_row = df[df["is_coincident_peak"]]
    peak_hour = int(peak_row["hour"].iloc[0]) if len(peak_row) else None
    plt.figure(figsize=(7, 4))
    plt.plot(df["hour"], df["citywide_kwh"], marker="o", color="#ff6b35")
    if peak_hour is not None:
        plt.axvline(peak_hour, color="gray", linestyle="--", linewidth=1,
                    label=f"Coincident peak hour ({peak_hour}:00)")
        plt.legend()
    plt.xlabel("Hour of day")
    plt.ylabel("City-wide demand (kWh)")
    plt.title("City-wide Hourly EV Charging Demand (dual-peak model)")
    savefig("fig1c_hourly_citywide_demand.png")


def fig_convergence():
    df = pd.read_csv(rpath("table_convergence.csv"))
    # Column -> (label, color). Only columns actually present get plotted
    # AND named in the title/legend — this mapping is the single source of
    # truth for both, so they cannot disagree with each other.
    series_spec = {
        "cpu_fitness": ("CPU PSO", "#4c9eff"),
        "gpu_fitness": ("CUDA PSO", "#ff6b35"),
        "ga_fitness": ("GA", "#a259ff"),
    }
    present = [col for col in series_spec if col in df.columns]

    plt.figure(figsize=(6, 4))
    for col in present:
        label, color = series_spec[col]
        plt.plot(df["iteration"], df[col], label=label, color=color)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Fitness")
    plt.title("Convergence Curve — " + " vs ".join(series_spec[c][0] for c in present))
    plt.legend()
    savefig("fig2_convergence.png")


def _gpu_suffix(df):
    if "gpu_available" in df.columns and not bool(df["gpu_available"].iloc[0]):
        return " (CPU fallback — no GPU detected)"
    return ""


def fig_speedup():
    df = pd.read_csv(rpath("table_speedup_vs_particles.csv"))
    plt.figure(figsize=(6, 4))
    plt.plot(df["n_particles"], df["speedup"], marker="o", color="#e8ff47")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Number of Particles")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title("Speedup vs Swarm Size" + _gpu_suffix(df))
    savefig("fig3_speedup.png")


def fig_speedup_vs_dataset_size():
    df = pd.read_csv(rpath("table_speedup_vs_dataset_size.csv"))
    plt.figure(figsize=(6, 4))
    plt.plot(df["n_sites"], df["speedup"], marker="o", color="#ff6b35")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Number of Wards (dataset size)")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title("Speedup vs Dataset Size" + _gpu_suffix(df))
    savefig("fig3b_speedup_vs_dataset_size.png")


def fig_pareto():
    df = pd.read_csv(rpath("table_pareto.csv"))
    plt.figure(figsize=(6, 4))
    # Scatter only — NO connecting line. Each point is an independently
    # evaluated, non-dominated station layout; nothing was evaluated
    # between them, so a line would imply a continuum that doesn't exist.
    plt.scatter(df["cost"], df["coverage_loss"], color="#36d399", s=45, zorder=3)
    plt.xlabel("Cost (infrastructure + distance + overload)")
    plt.ylabel("Coverage Loss")
    plt.title(f"Pareto Archive — Non-Dominated Solutions (N={len(df)})")
    savefig("fig4_pareto.png")


def fig_map():
    df = pd.read_csv(rpath("table0_wards.csv"))
    plt.figure(figsize=(7, 6))
    plt.scatter(df["lon"], df["lat"], c="#4c9eff", s=25)
    for _, row in df.iterrows():
        plt.annotate(row["ward"], (row["lon"], row["lat"]), fontsize=5, alpha=0.8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"BBMP Ward Locations")
    savefig("fig5_map.png")


def fig_ablation():
    df = pd.read_csv(rpath("ablation.csv"))
    n_runs = int(df["N_Runs"].iloc[0]) if "N_Runs" in df.columns else None
    plt.figure(figsize=(6, 4))
    plt.bar(df["Version"], df["Fitness_Mean"], yerr=df["Fitness_Std"], capsize=4,
            color=["#4c9eff", "#e8ff47", "#ff6b35"])
    plt.ylabel("Fitness (lower = better)")
    title = "Ablation: Demand Model Comparison"
    if n_runs:
        title += f" (mean ± std, N={n_runs} runs)"
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    savefig("fig6_ablation.png")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating figures from computed results (no synthetic data, no live "
          "core.py calls — every number below is read from a CSV)...")
    fig_demand()
    fig_dual_peak()
    fig_hourly_curve()
    fig_convergence()
    fig_speedup()
    fig_speedup_vs_dataset_size()
    fig_pareto()
    fig_map()
    fig_ablation()
    print("Done.")
