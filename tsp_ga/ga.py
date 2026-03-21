from dataclasses import dataclass
import random
import time
from typing import Sequence

from .tsp import TSPInstance, build_distance_matrix, tour_distance


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 200
    generations: int = 600
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 5
    elitism: int = 2
    seed: int | None = None


# Freeze result objects so returned runs are immutable snapshots
@dataclass(frozen=True)
class GAResult:
    best_route: tuple[int, ...]
    best_distance: float
    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    best_generation: int
    runtime_seconds: float


def _ordered_crossover(
    parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random
) -> list[int]:
    size = len(parent_a)
    left, right = sorted(rng.sample(range(size), 2))
    child = [-1] * size
    child[left : right + 1] = parent_a[left : right + 1]
    in_child = set(child[left : right + 1])

    cursor = (right + 1) % size
    index_b = (right + 1) % size
    while -1 in child:
        city = parent_b[index_b]
        if city not in in_child:
            child[cursor] = city
            in_child.add(city)
            cursor = (cursor + 1) % size
        index_b = (index_b + 1) % size
    return child


def _mutate_inversion(route: list[int], rng: random.Random) -> None:
    left, right = sorted(rng.sample(range(len(route)), 2))
    route[left : right + 1] = reversed(route[left : right + 1])


def _tournament_select(
    ranked_population: Sequence[tuple[float, tuple[int, ...]]],
    tournament_size: int,
    rng: random.Random,
) -> tuple[int, ...]:
    participants = rng.sample(ranked_population, tournament_size)
    best = min(participants, key=lambda item: item[0])
    return best[1]


def _rank_population(
    population: Sequence[tuple[int, ...]], dist_matrix: Sequence[Sequence[float]]
) -> list[tuple[float, tuple[int, ...]]]:
    ranked = [(tour_distance(route, dist_matrix), route) for route in population]
    ranked.sort(key=lambda item: item[0])
    return ranked


def solve_tsp_ga(instance: TSPInstance, config: GAConfig | None = None) -> GAResult:
    if config is None:
        config = GAConfig()

    rng = random.Random(config.seed)
    city_ids = tuple(range(instance.size))
    population: list[tuple[int, ...]] = [
        tuple(rng.sample(city_ids, instance.size)) for _ in range(config.population_size)
    ]
    dist_matrix = build_distance_matrix(instance)

    history_best: list[float] = []
    history_mean: list[float] = []
    best_distance = float("inf")
    best_route: tuple[int, ...] | None = None
    best_generation = 0

    start_time = time.perf_counter()
    for generation in range(config.generations + 1):
        ranked = _rank_population(population, dist_matrix)
        distances = [item[0] for item in ranked]    
        current_best_distance = distances[0]
        current_best_route = ranked[0][1]
        current_mean_distance = sum(distances) / len(distances)

        history_best.append(current_best_distance)
        history_mean.append(current_mean_distance)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route
            best_generation = generation

        if generation == config.generations:
            break

        next_population: list[tuple[int, ...]] = [
            ranked[idx][1] for idx in range(config.elitism)
        ]
        while len(next_population) < config.population_size:
            parent_a = _tournament_select(ranked, config.tournament_size, rng)
            parent_b = _tournament_select(ranked, config.tournament_size, rng)

            if rng.random() < config.crossover_rate:
                child = _ordered_crossover(parent_a, parent_b, rng)
            else:
                child = list(parent_a)

            if rng.random() < config.mutation_rate:
                _mutate_inversion(child, rng)

            next_population.append(tuple(child))

        population = next_population

    runtime_seconds = time.perf_counter() - start_time
    assert best_route is not None
    return GAResult(
        best_route=best_route,
        best_distance=best_distance,
        history_best=tuple(history_best),
        history_mean=tuple(history_mean),
        best_generation=best_generation,
        runtime_seconds=runtime_seconds,
    )
