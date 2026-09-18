
"""
gpu.py
======
Real CUDA kernel for fitness evaluation, implementing the *exact same*
formula as core.compute_fitness (coverage / distance / overload — paper
Eq. 2). This matters: the previous CUDA kernel computed a completely
different quantity (mean nearest-neighbour distance only, Euclidean on
raw lat/lon degrees) than the CPU path, so any CPU-vs-GPU "fitness"
comparison in the old code was comparing two different functions, not
CPU vs GPU performance on the same objective.

This file requires an actual CUDA-capable GPU + the CUDA toolkit to run
the kernel. On a machine without a GPU, cuda_is_available() returns False
and core.run_cuda_pso() falls back to the CPU implementation with an
explicit printed warning — it will NOT silently report a fake speedup.
"""

import math
import numpy as np

try:
    from numba import cuda
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

EARTH_RADIUS_KM = 6371.0
MAX_K = 32  # local-array upper bound for stations per particle on device


def cuda_is_available():
    if not _NUMBA_OK:
        return False
    try:
        return cuda.is_available()
    except Exception:
        return False


if _NUMBA_OK:

    @cuda.jit(device=True, inline=True)
    def _haversine_km(lat1, lon1, lat2, lon2):
        p1 = lat1 * math.pi / 180.0
        p2 = lat2 * math.pi / 180.0
        dphi = (lat2 - lat1) * math.pi / 180.0
        dlmb = (lon2 - lon1) * math.pi / 180.0
        a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
        a = min(1.0, max(0.0, a))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return EARTH_RADIUS_KM * c

    @cuda.jit
    def _fitness_kernel(pos, fitness, sites, demand, n_sites, k,
                         coverage_radius, grid_cap, alpha, beta, gamma, max_dist_norm):
        i = cuda.grid(1)
        if i >= pos.shape[0]:
            return

        idxs = cuda.local.array(MAX_K, dtype=numba_int32)
        for j in range(k):
            v = pos[i, j]
            idx = int(v * n_sites) % n_sites
            collision = True
            while collision:
                collision = False
                for jj in range(j):
                    if idxs[jj] == idx:
                        collision = True
                        idx = (idx + 1) % n_sites
                        break
            idxs[j] = idx

        station_load = cuda.local.array(MAX_K, dtype=numba_float64)
        for j in range(k):
            station_load[j] = 0.0

        total_demand = 0.0
        covered_demand = 0.0
        weighted_dist = 0.0

        for s in range(n_sites):
            min_d = 1.0e18
            min_j = 0
            for j in range(k):
                st = idxs[j]
                d = _haversine_km(sites[s, 0], sites[s, 1], sites[st, 0], sites[st, 1])
                if d < min_d:
                    min_d = d
                    min_j = j

            dem = demand[s]
            total_demand += dem
            weighted_dist += min_d * dem
            if min_d <= coverage_radius:
                covered_demand += dem
            station_load[min_j] += dem

        coverage = covered_demand / total_demand
        mean_dist_norm = (weighted_dist / total_demand) / max_dist_norm

        overload = 0.0
        for j in range(k):
            excess = station_load[j] - grid_cap
            if excess > 0.0:
                overload += excess
        overload = overload / (k * grid_cap)

        fitness[i] = alpha * (1.0 - coverage) + beta * mean_dist_norm + gamma * overload

    import numba
    numba_int32 = numba.int32
    numba_float64 = numba.float64


def evaluate_batch_gpu(pos, sites_arr, demand, k_stations):
    """
    pos: (n_particles, k_stations) float64 in [0,1]
    Returns fitness array (n_particles,) computed on the GPU with the kernel
    above. Only call this after checking cuda_is_available(). Used by
    verify_consistency.py for a one-shot CPU-vs-GPU fitness check; the
    actual optimization loop uses run_pso_gpu_resident below instead of
    calling this once per iteration.
    """
    from backend.core import (COVERAGE_RADIUS_KM, GRID_CAP_KW, ALPHA, BETA, GAMMA,
                               MAX_DIST_NORM_KM)

    n_particles = pos.shape[0]
    n_sites = sites_arr.shape[0]

    d_pos = cuda.to_device(np.ascontiguousarray(pos, dtype=np.float64))
    d_sites = cuda.to_device(np.ascontiguousarray(sites_arr, dtype=np.float64))
    d_demand = cuda.to_device(np.ascontiguousarray(demand, dtype=np.float64))
    d_fit = cuda.device_array(n_particles, dtype=np.float64)

    threads = 128
    blocks = (n_particles + threads - 1) // threads

    _fitness_kernel[blocks, threads](
        d_pos, d_fit, d_sites, d_demand, n_sites, k_stations,
        COVERAGE_RADIUS_KM, GRID_CAP_KW, ALPHA, BETA, GAMMA, MAX_DIST_NORM_KM,
    )
    cuda.synchronize()
    return d_fit.copy_to_host()


# =====================================================================
# GPU-RESIDENT PSO LOOP
# =====================================================================
# The previous version of this file only put the FITNESS evaluation on the
# GPU: run_cuda_pso() still did the velocity/position update in numpy on
# the CPU every iteration, and re-uploaded the entire (n_particles,
# k_stations) position matrix to the device on every single call to
# evaluate_batch_gpu. For the swarm sizes this paper actually reports on,
# that per-iteration host<->device round trip of the whole swarm is a real
# cost competing with the tiny amount of compute being offloaded — it is a
# very plausible part of why Table II's original speedup was only ~1.008
# (the paper's own Section V.B already blames "some CPU involvement in
# fitness calculation", but the update step was ALSO on the CPU, which
# that sentence didn't account for).
#
# This version keeps position, velocity, pbest, pbest_val, sites, and
# demand resident on the device for the ENTIRE run. Per iteration, only
# two small arrays cross the PCIe bus: pbest_val (n_particles floats, to
# find the argmin on the host) and — only when gbest actually improves —
# a single (k_stations,) row. The velocity/position update itself runs in
# a CUDA kernel using an on-device xoroshiro128p RNG (numba.cuda.random),
# not numpy, so no random numbers need to be generated on the host either.
if _NUMBA_OK:
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

    @cuda.jit
    def _pso_update_kernel(pos, vel, pbest, gbest, rng_states, w, c1, c2, k):
        i = cuda.grid(1)
        if i >= pos.shape[0]:
            return
        for j in range(k):
            r1 = xoroshiro128p_uniform_float64(rng_states, i)
            r2 = xoroshiro128p_uniform_float64(rng_states, i)
            newvel = (w * vel[i, j]
                      + c1 * r1 * (pbest[i, j] - pos[i, j])
                      + c2 * r2 * (gbest[j] - pos[i, j]))
            newpos = pos[i, j] + newvel
            if newpos < 0.0:
                newpos = 0.0
            elif newpos > 1.0:
                newpos = 1.0
            vel[i, j] = newvel
            pos[i, j] = newpos

    @cuda.jit
    def _pso_pbest_update_kernel(fit, pbest_val, pos, pbest, k):
        i = cuda.grid(1)
        if i >= fit.shape[0]:
            return
        if fit[i] < pbest_val[i]:
            pbest_val[i] = fit[i]
            for j in range(k):
                pbest[i, j] = pos[i, j]


def run_pso_gpu_resident(sites_arr, demand, n_particles=50, k_stations=6, max_iter=80,
                          w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=None):
    """
    Fully GPU-resident PSO: update, fitness evaluation, and pbest
    maintenance all happen in CUDA kernels with pos/vel/pbest/pbest_val
    kept on the device across the whole run. Returns
    (gbest, gbest_val, history, stations) — same shape as run_cpu_pso /
    the old run_cuda_pso, so callers don't need to change.
    """
    from backend.core import (COVERAGE_RADIUS_KM, GRID_CAP_KW, ALPHA, BETA, GAMMA,
                               MAX_DIST_NORM_KM, decode_particle)

    k_stations = min(k_stations, sites_arr.shape[0])
    n_sites = sites_arr.shape[0]

    rng_np = np.random.default_rng(seed)
    pos0 = rng_np.random((n_particles, k_stations))

    d_pos = cuda.to_device(np.ascontiguousarray(pos0))
    d_vel = cuda.to_device(np.zeros_like(pos0))
    d_pbest = cuda.to_device(pos0.copy())
    d_sites = cuda.to_device(np.ascontiguousarray(sites_arr, dtype=np.float64))
    d_demand = cuda.to_device(np.ascontiguousarray(demand, dtype=np.float64))
    d_fit = cuda.device_array(n_particles, dtype=np.float64)

    threads = 128
    blocks = (n_particles + threads - 1) // threads

    seed_val = int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
    rng_states = create_xoroshiro128p_states(n_particles, seed=seed_val)

    # Initial fitness pass seeds pbest/pbest_val.
    _fitness_kernel[blocks, threads](
        d_pos, d_fit, d_sites, d_demand, n_sites, k_stations,
        COVERAGE_RADIUS_KM, GRID_CAP_KW, ALPHA, BETA, GAMMA, MAX_DIST_NORM_KM,
    )
    cuda.synchronize()
    d_pbest_val = cuda.to_device(d_fit.copy_to_host())

    pbest_val_host = d_pbest_val.copy_to_host()
    gbest_idx = int(np.argmin(pbest_val_host))
    gbest_val = float(pbest_val_host[gbest_idx])
    gbest_row = d_pbest[gbest_idx:gbest_idx + 1, :].copy_to_host()[0]
    d_gbest = cuda.to_device(gbest_row)

    history = [gbest_val]

    for it in range(max_iter):
        w = w_start - (w_start - w_end) * (it / max(1, max_iter - 1))

        _pso_update_kernel[blocks, threads](d_pos, d_vel, d_pbest, d_gbest, rng_states, w, c1, c2, k_stations)
        _fitness_kernel[blocks, threads](
            d_pos, d_fit, d_sites, d_demand, n_sites, k_stations,
            COVERAGE_RADIUS_KM, GRID_CAP_KW, ALPHA, BETA, GAMMA, MAX_DIST_NORM_KM,
        )
        _pso_pbest_update_kernel[blocks, threads](d_fit, d_pbest_val, d_pos, d_pbest, k_stations)
        cuda.synchronize()

        pbest_val_host = d_pbest_val.copy_to_host()  # small: n_particles floats
        it_best = int(np.argmin(pbest_val_host))
        if pbest_val_host[it_best] < gbest_val:
            gbest_val = float(pbest_val_host[it_best])
            gbest_row = d_pbest[it_best:it_best + 1, :].copy_to_host()[0]  # small: k floats
            d_gbest = cuda.to_device(gbest_row)

        history.append(gbest_val)

    gbest_final = d_gbest.copy_to_host()
    stations = decode_particle(gbest_final, sites_arr, k_stations)
    return gbest_final, gbest_val, history, stations


if __name__ == "__main__":
    print("numba installed:", _NUMBA_OK)
    print("CUDA device available:", cuda_is_available())
