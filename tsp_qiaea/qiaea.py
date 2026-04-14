from dataclasses import dataclass
import time
import math
from typing import Sequence
import numpy as np
from numba import njit, prange
from .tsp import TSPInstance, build_distance_matrix

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

@dataclass(frozen=True)
class QIAEAConfig:
    population_size: int = 80
    generations: int = 500
    n_males: int = 3
    elitism: int = 2
    substring_fraction: float = 0.20
    substring_fraction_end: float | None = None
    bits_per_city: int = 16  # Kept for API compatibility; overridden internally
    random_tie_break: bool = True
    seed: int | None = None
    stagnation_patience: int = 15

@dataclass(frozen=True)
class QIAEAResult:
    best_route: tuple[int, ...]
    best_distance: float
    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    best_generation: int
    runtime_seconds: float


@njit(parallel=True)
def _measure_and_collapse(quantum_pop, rand_vals, measured_bits, collapsed_pop):
    """Measure each qubit and collapse amplitudes into a classical bit population"""
    pop_size, n_bits, _ = quantum_pop.shape
    for i in prange(pop_size):
        for j in range(n_bits):
            p0 = quantum_pop[i, j, 0] ** 2
            p0 = min(max(p0, 0.0), 1.0)
            bit = 1 if rand_vals[i, j] > p0 else 0
            measured_bits[i, j] = bit
            if bit == 0:
                collapsed_pop[i, j, 0] = 1.0
                collapsed_pop[i, j, 1] = 0.0
            else:
                collapsed_pop[i, j, 0] = 0.0
                collapsed_pop[i, j, 1] = 1.0


@njit(parallel=True)
def _decode_routes_parallel(measured_bits, n_cities, start_city, dist_matrix, routes):
    """Decode measured adjacency bits into valid closed tours with greedy repair"""
    pop_size = measured_bits.shape[0]
    for i in prange(pop_size):
        route = np.empty(n_cities + 1, dtype=np.int64)
        route[0] = start_city
        visited = np.zeros(n_cities, dtype=np.bool_)
        visited[start_city] = True
        current = start_city

        for step in range(1, n_cities):
            best_city = -1
            best_dist = 1e18
            has_preferred = False
            
            # Scan unvisited cities for edges marked '1' by quantum measurement
            for j in range(n_cities):
                if not visited[j]:
                    if measured_bits[i, current * n_cities + j] == 1:
                        has_preferred = True
                        d = dist_matrix[current, j]
                        if d < best_dist:
                            best_dist = d
                            best_city = j

            if has_preferred:
                next_city = best_city
            else:
                # Fallback: nearest unvisited city (guarantees valid permutation)
                for j in range(n_cities):
                    if not visited[j]:
                        d = dist_matrix[current, j]
                        if d < best_dist:
                            best_dist = d
                            best_city = j
                next_city = best_city

            route[step] = next_city
            visited[next_city] = True
            current = next_city

        route[n_cities] = start_city  # Close loop
        routes[i] = route


@njit(parallel=True)
def _calc_distances(routes, dist_matrix, distances):
    """Compute full tour distance for every route in parallel"""
    pop_size = routes.shape[0]
    n_cities = routes.shape[1] - 1
    for i in prange(pop_size):
        total = 0.0
        for j in range(n_cities):
            total += dist_matrix[routes[i, j], routes[i, j + 1]]
        distances[i] = total


@njit(parallel=True)
def _crossover_offspring_parallel(
    queen_quantum, queen_bits, male_bits_pool, rand_starts, rand_lens, rand_males,
    next_population, elitism, pop_size
):
    """Create offspring by combining queen and male bit patterns with quantum gates"""
    n_bits = queen_quantum.shape[0]
    for idx in prange(elitism, pop_size):
        # Deterministic male selection from pre-generated array
        male_idx = rand_males[idx - elitism]
        male_bits = male_bits_pool[male_idx]
        
        # Clone queen quantum state
        for j in range(n_bits):
            next_population[idx, j, 0] = queen_quantum[j, 0]
            next_population[idx, j, 1] = queen_quantum[j, 1]

        # Determine crossover window
        start = rand_starts[idx - elitism]
        length = rand_lens[idx - elitism]
        end = start + length

        # Apply quantum gates based on bit comparison
        for j in range(start, end):
            if queen_bits[j] == male_bits[j]:
                # Match -> Hadamard gate
                a, b = next_population[idx, j, 0], next_population[idx, j, 1]
                next_population[idx, j, 0] = (a + b) * _INV_SQRT2
                next_population[idx, j, 1] = (a - b) * _INV_SQRT2
            else:
                # Mismatch -> Pauli-X gate (swap amplitudes)
                a, b = next_population[idx, j, 0], next_population[idx, j, 1]
                next_population[idx, j, 0] = b
                next_population[idx, j, 1] = a

        # Normalize modified segment
        for j in range(start, end):
            norm = math.sqrt(next_population[idx, j, 0]**2 + next_population[idx, j, 1]**2)
            if norm > 1e-12:
                next_population[idx, j, 0] /= norm
                next_population[idx, j, 1] /= norm


def _validate_route(route: Sequence[int], n_cities: int) -> None:
    """Ensure decoded route is a complete closed permutation without duplicates"""
    seen = set(route[:-1])
    if len(route) != n_cities + 1 or len(seen) != n_cities:
        raise RuntimeError("QIAEA produced an invalid route (not a full permutation).")
    if route[0] != route[-1]:
        raise RuntimeError("QIAEA route is not closed (start != end).")


def solve_tsp_qiaea(
    instance: TSPInstance,
    config: QIAEAConfig | None = None,
    *,
    start_city: int = 0,
) -> QIAEAResult:
    """Run the full QIAEA loop and return best route with convergence history"""
    if config is None:
        config = QIAEAConfig()
        
    n_cities = instance.size
    if n_cities < 2:
        raise ValueError("TSP instance must have at least 2 cities.")
    if not (0 <= start_city < n_cities):
        raise ValueError("start_city must be in [0, n_cities).")

    # TSP Adaptation: N*N adjacency matrix encoding
    chromosome_bits = n_cities * n_cities
    
    population_size = max(4, int(config.population_size))
    max_elitism = population_size - 1
    elitism = max(1, min(int(config.elitism), max_elitism))
    n_males = max(1, min(int(config.n_males), population_size - 1))
    
    substring_fraction_start = float(config.substring_fraction)
    substring_fraction_end = (
        substring_fraction_start
        if config.substring_fraction_end is None
        else float(config.substring_fraction_end)
    )
    stagnation_limit = max(1, int(config.stagnation_patience))

    rng = np.random.default_rng(config.seed)
    dist_matrix = np.ascontiguousarray(np.asarray(build_distance_matrix(instance), dtype=np.float64))

    # Pre-allocate memory for Numba kernels
    quantum_population = np.empty((population_size, chromosome_bits, 2), dtype=np.float64)
    quantum_population[..., 0] = _INV_SQRT2
    quantum_population[..., 1] = _INV_SQRT2
    
    measured_bits = np.empty((population_size, chromosome_bits), dtype=np.int64)
    collapsed_population = np.empty_like(quantum_population)
    routes = np.empty((population_size, n_cities + 1), dtype=np.int64)
    distances = np.empty(population_size, dtype=np.float64)
    
    # Pre-generate random arrays for thread-safe parallel execution
    rand_measure = rng.random((population_size, chromosome_bits))
    rand_starts = np.empty(population_size - elitism, dtype=np.int64)
    rand_lens = np.empty(population_size - elitism, dtype=np.int64)
    rand_males = np.empty(population_size - elitism, dtype=np.int64)

    history_best: list[float] = []
    history_mean: list[float] = []

    best_distance = float("inf")
    best_route: tuple[int, ...] | None = None
    best_generation = 0
    stagnation_counter = 0

    start_time = time.perf_counter()

    for generation in range(config.generations + 1):
        # 1. Parallel Measurement & Collapse
        _measure_and_collapse(quantum_population, rand_measure, measured_bits, collapsed_population)
        
        # 2. Parallel Decoding & Distance Evaluation
        _decode_routes_parallel(measured_bits, n_cities, start_city, dist_matrix, routes)
        _calc_distances(routes, dist_matrix, distances)
        
        ranked_indices = np.argsort(distances, kind="stable")

        current_best_distance = float(distances[ranked_indices[0]])
        history_best.append(current_best_distance)
        history_mean.append(float(np.mean(distances)))

        # 3. Update Global Best
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            route = tuple(int(city) for city in routes[ranked_indices[0]])
            _validate_route(route, n_cities)
            best_route = route
            best_generation = generation
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if generation == config.generations:
            break

        # 4. Colony Structure {Q, M, R}
        queen_idx = int(ranked_indices[0])
        queen_bits = measured_bits[queen_idx]
        queen_quantum = collapsed_population[queen_idx]
        male_indices = ranked_indices[1 : 1 + n_males]
        male_bits_pool = measured_bits[male_indices]

        # 5. Disaster Operator (Paper p.6)
        if stagnation_counter >= stagnation_limit:
            quantum_population[..., 0] = _INV_SQRT2
            quantum_population[..., 1] = _INV_SQRT2
            stagnation_counter = 0
            # Refresh random arrays for next gen
            rand_measure[:] = rng.random((population_size, chromosome_bits))

        # 6. Parallel Crossover for Rest (R) members
        next_population = np.empty_like(quantum_population)
        next_population[:elitism] = collapsed_population[ranked_indices[:elitism]]
        
        substring_len = int(round(
            (substring_fraction_start + (substring_fraction_end - substring_fraction_start) * (generation / config.generations)) * chromosome_bits
        ))
        substring_len = max(1, min(substring_len, chromosome_bits))
        
        # Pre-generate crossover random parameters
        rand_starts[:] = rng.integers(0, chromosome_bits - substring_len + 1, size=population_size - elitism)
        rand_lens[:] = substring_len
        rand_males[:] = rng.integers(0, n_males, size=population_size - elitism)
        
        _crossover_offspring_parallel(
            queen_quantum, queen_bits, male_bits_pool, rand_starts, rand_lens, rand_males,
            next_population, elitism, population_size
        )
        quantum_population = next_population
        # Refresh measurement randomness for next generation
        rand_measure[:] = rng.random((population_size, chromosome_bits))

    runtime_seconds = time.perf_counter() - start_time
    if best_route is None:
        raise RuntimeError("QIAEA finished without producing a valid route.")

    return QIAEAResult(
        best_route=best_route,
        best_distance=best_distance,
        history_best=tuple(history_best),
        history_mean=tuple(history_mean),
        best_generation=best_generation,
        runtime_seconds=runtime_seconds,
    )


# Backward compatibility for older imports.
solve_tsp_qiaea_mat = solve_tsp_qiaea

