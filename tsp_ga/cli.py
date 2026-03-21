import argparse
from pathlib import Path
import sys

from .ga import GAConfig, solve_tsp_ga
from .io import load_tsplib
from .tsp import TSPInstance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=600)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.tsp:
        instance = load_tsplib(args.tsp)
    else:
        parser.error("Use --tsp <file>")
        return 2

    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        elitism=args.elitism,
        seed=args.seed,
    )

    result = solve_tsp_ga(instance, config)
    route_1based = [city + 1 for city in result.best_route]

    print(f"Instance: {instance.name} ({instance.size} cities)")
    print(
        "Algorithm: GA | "
        f"population={config.population_size}, generations={config.generations}, "
        f"crossover_rate={config.crossover_rate}, mutation_rate={config.mutation_rate}, "
        f"tournament_size={config.tournament_size}, elitism={config.elitism}, seed={config.seed}"
    )
    print(f"Best distance: {result.best_distance:.6f}")
    print("Best route (1-based): " + " -> ".join(map(str, route_1based + [route_1based[0]])))
    print(
        f"Convergence: start_best={result.history_best[0]:.6f}, "
        f"end_best={result.history_best[-1]:.6f}, "
        f"best_generation={result.best_generation}, runtime_s={result.runtime_seconds:.4f}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
