import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.core import sites, generate_demand, compute_fitness, decode_particle
    from backend.gpu import cuda_is_available, evaluate_batch_gpu
except ImportError:
    from core import sites, generate_demand, compute_fitness, decode_particle
    from gpu import cuda_is_available, evaluate_batch_gpu


def main():
    if not cuda_is_available():
        print("No CUDA device detected — cannot verify the GPU kernel here.")
        return

    np.random.seed(123)
    demand = generate_demand(seed=123)
    k_stations = 6
    n_particles = 500

    pos = np.random.rand(n_particles, k_stations)

    cpu_fit = np.array([
        compute_fitness(decode_particle(pos[i], sites, k_stations), sites, demand)
        for i in range(n_particles)
    ])
    gpu_fit = evaluate_batch_gpu(pos, sites, demand, k_stations)

    abs_diff = np.abs(cpu_fit - gpu_fit)
    print(f"n_particles checked: {n_particles}")
    print(f"max abs difference:  {abs_diff.max():.3e}")
    print(f"mean abs difference: {abs_diff.mean():.3e}")

    tol = 1e-6
    if abs_diff.max() < tol:
        print(f"PASS: CPU and GPU fitness match within {tol:.0e} for all {n_particles} particles.")
    else:
        print(f"FAIL: CPU and GPU fitness differ by more than {tol:.0e}.")


if __name__ == "__main__":
    main()
