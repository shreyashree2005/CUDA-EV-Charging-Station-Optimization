

"""
ablation.py
===========
Table III in the paper claims three distinct demand-modeling conditions
(uniform / demand-based / "advanced" time-series). The previous version of
this file generated V3 by calling run_cpu_pso on the *exact same* `demand`
array used for V2 — see the comment that was literally in that file:
"# V3: Same but call it advanced". V2 and V3 were mathematically guaranteed
to differ only by PSO's own randomness, not by any change in the model.

This version runs three genuinely different demand models, all defined in
core.generate_demand():
  V1 uniform  — every ward has identical demand (no spatial signal at all)
  V2 density  — demand proportional to population density (spatial only)
  V3 temporal — density scaled by the dual-peak daily profile from
                Section IV-B (spatial + temporal)

Each condition is run n_runs times (fresh random PSO seed each time, same
demand-generation seed within a condition so the demand array itself is
held fixed while the algorithm's own stochasticity is averaged out), and
we report mean +/- std, exactly like Table II does for CPU vs GPU.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    from backend.core import sites, generate_demand, run_cpu_pso
except ImportError:
    from core import sites, generate_demand, run_cpu_pso

N_RUNS = 5
N_PARTICLES = 60
MAX_ITER = 60
K_STATIONS = 6
DEMAND_SEED = 42  # fixed so V1/V2/V3 are compared under one consistent demand draw

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_condition(name, demand_model):
    demand = generate_demand(seed=DEMAND_SEED, demand_model=demand_model)
    fitnesses, times = [], []
    for run in range(N_RUNS):
        np.random.seed(1000 + run)  # vary PSO stochasticity across runs
        t0 = time.time()
        _, fit, _, _ = run_cpu_pso(sites, demand, n_particles=N_PARTICLES,
                                    k_stations=K_STATIONS, max_iter=MAX_ITER)
        times.append(time.time() - t0)
        fitnesses.append(fit)
    return {
        "Version": name,
        "demand_model": demand_model,
        "Fitness_Mean": float(np.mean(fitnesses)),
        "Fitness_Std": float(np.std(fitnesses)),
        "Time_Mean_s": float(np.mean(times)),
        "Demand_Total_kWh": float(demand.sum()),
        "N_Runs": N_RUNS,
    }


if __name__ == "__main__":
    print("Running ablation study (3 genuinely distinct demand models,"
          f" {N_RUNS} runs each)...")

    rows = [
        run_condition("V1 Uniform Demand", "uniform"),
        run_condition("V2 Density-Based Demand", "density"),
        run_condition("V3 Dual-Peak Temporal Demand", "temporal"),
    ]

    for r in rows:
        print(f"  {r['Version']}: fitness = {r['Fitness_Mean']:.4f} "
              f"(+/- {r['Fitness_Std']:.4f}), time = {r['Time_Mean_s']:.2f}s")

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "ablation.csv")
    df.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}")
