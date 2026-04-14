from dataclasses import dataclass
import time

import numpy as np

from .tsp import TSPInstance, build_distance_matrix


@dataclass(frozen=True)
class ACOConfig:
    population_size: int = 20
    generations: int = 120
    alpha: float = 1.0
    beta: float = 3.0
    evaporation_rate: float = 0.25
    pheromone_deposit: float = 100.0
    elitist_weight: float = 2.0
    min_pheromone: float = 1e-6
    max_pheromone: float = 1e6
    seed: int | None = None


@dataclass(frozen=True)
class ACOResult:
    best_route: tuple[int, ...]
    best_distance: float
    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    best_generation: int
    runtime_seconds: float


def _build_ant_route(
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    *,
    start_city: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct one ant tour by probabilistic city selection with pheromone guidance"""
    n_cities = pheromone.shape[0]
    route = np.empty(n_cities, dtype=np.int64)
    visited = np.zeros(n_cities, dtype=np.bool_)

    route[0] = start_city
    visited[start_city] = True
    current = start_city

    for step in range(1, n_cities):
        candidates = np.flatnonzero(~visited)

        tau = np.power(pheromone[current, candidates], alpha, dtype=np.float64)
        eta = np.power(heuristic[current, candidates], beta, dtype=np.float64)
        weights = tau * eta
        total = float(np.sum(weights))

        if (not np.isfinite(total)) or total <= 0.0:
            candidate_idx = int(rng.integers(0, len(candidates)))
            next_city = int(candidates[candidate_idx])
        else:
            probabilities = weights / total
            cumulative = np.cumsum(probabilities)
            probe = float(rng.random())
            selected = int(np.searchsorted(cumulative, probe, side="right"))
            if selected >= len(candidates):
                selected = len(candidates) - 1
            next_city = int(candidates[selected])

        route[step] = next_city
        visited[next_city] = True
        current = next_city

    return route


def _tour_distance(route: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Compute closed route distance from a numpy route vector"""
    next_route = np.roll(route, -1)
    return float(np.sum(dist_matrix[route, next_route]))


def _deposit_route_pheromone(
    pheromone: np.ndarray, route: np.ndarray, amount: float
) -> None:
    """Deposit pheromone along each edge of a route in both directions"""
    n_cities = route.shape[0]
    for idx in range(n_cities):
        a = int(route[idx])
        b = int(route[(idx + 1) % n_cities])
        pheromone[a, b] += amount
        pheromone[b, a] += amount


def solve_tsp_aco(
    instance: TSPInstance,
    config: ACOConfig | None = None,
    *,
    start_city: int = 0,
) -> ACOResult:
    """Run the full ant colony loop and return best route with convergence history"""
    if config is None:
        config = ACOConfig()

    n_cities = instance.size
    if n_cities < 3:
        raise ValueError("TSP instance must have at least 3 cities.")
    if not (0 <= start_city < n_cities):
        raise ValueError("start_city must be in [0, n_cities).")

    population_size = max(2, int(config.population_size))
    generations = max(0, int(config.generations))
    alpha = max(0.0, float(config.alpha))
    beta = max(0.0, float(config.beta))
    evaporation_rate = min(max(float(config.evaporation_rate), 0.0), 0.999999)
    pheromone_deposit = max(float(config.pheromone_deposit), 1e-12)
    elitist_weight = max(float(config.elitist_weight), 0.0)
    min_pheromone = max(float(config.min_pheromone), 0.0)
    max_pheromone = max(float(config.max_pheromone), min_pheromone + 1e-12)

    rng = np.random.default_rng(config.seed)
    dist_matrix = np.asarray(build_distance_matrix(instance), dtype=np.float64)
    heuristic = np.divide(
        1.0,
        dist_matrix,
        out=np.zeros_like(dist_matrix),
        where=dist_matrix > 0.0,
    )
    np.fill_diagonal(heuristic, 0.0)

    off_diag = dist_matrix[dist_matrix > 0.0]
    mean_distance = float(np.mean(off_diag)) if off_diag.size else 1.0
    initial_pheromone = 1.0 / max(mean_distance * n_cities, 1e-12)
    pheromone = np.full((n_cities, n_cities), initial_pheromone, dtype=np.float64)
    np.fill_diagonal(pheromone, 0.0)

    history_best: list[float] = []
    history_mean: list[float] = []

    best_distance = float("inf")
    best_route_array: np.ndarray | None = None
    best_generation = 0

    start_time = time.perf_counter()

    for generation in range(generations + 1):
        routes = np.empty((population_size, n_cities), dtype=np.int64)
        distances = np.empty(population_size, dtype=np.float64)

        for ant_idx in range(population_size):
            route = _build_ant_route(
                pheromone=pheromone,
                heuristic=heuristic,
                start_city=start_city,
                alpha=alpha,
                beta=beta,
                rng=rng,
            )
            routes[ant_idx] = route
            distances[ant_idx] = _tour_distance(route, dist_matrix)

        best_idx = int(np.argmin(distances))
        current_best = float(distances[best_idx])
        history_best.append(current_best)
        history_mean.append(float(np.mean(distances)))

        if current_best < best_distance:
            best_distance = current_best
            best_route_array = routes[best_idx].copy()
            best_generation = generation

        if generation == generations:
            break

        pheromone *= 1.0 - evaporation_rate
        if min_pheromone > 0.0:
            np.maximum(pheromone, min_pheromone, out=pheromone)

        for ant_idx in range(population_size):
            distance = float(distances[ant_idx])
            contribution = pheromone_deposit / max(distance, 1e-12)
            _deposit_route_pheromone(pheromone, routes[ant_idx], contribution)

        if best_route_array is not None and elitist_weight > 0.0:
            best_contribution = (pheromone_deposit * elitist_weight) / max(best_distance, 1e-12)
            _deposit_route_pheromone(pheromone, best_route_array, best_contribution)

        np.clip(pheromone, min_pheromone, max_pheromone, out=pheromone)
        np.fill_diagonal(pheromone, 0.0)

    runtime_seconds = time.perf_counter() - start_time
    if best_route_array is None:
        raise RuntimeError("ACO finished without producing a valid route.")

    return ACOResult(
        best_route=tuple(int(city) for city in best_route_array),
        best_distance=best_distance,
        history_best=tuple(history_best),
        history_mean=tuple(history_mean),
        best_generation=best_generation,
        runtime_seconds=runtime_seconds,
    )
