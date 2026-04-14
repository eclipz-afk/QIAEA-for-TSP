import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import statistics
from typing import Iterable

from .aco import ACOConfig, solve_tsp_aco
from .io import load_tsplib


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    path: Path
    optimum: float


@dataclass(frozen=True)
class InstanceStatistics:
    instance: str
    optimum: float
    runs: int
    best_distance_min: float
    best_distance_mean: float
    best_distance_std: float
    relative_error_mean_pct: float
    relative_error_std_pct: float
    accuracy_hit_rate_pct: float
    mean_best_generation: float
    mean_target_generation: float | None
    mean_runtime_seconds: float


@dataclass(frozen=True)
class OverallStatistics:
    runs_per_instance: int
    instances: int
    mean_relative_error_pct: float
    mean_accuracy_hit_rate_pct: float
    mean_runtime_seconds: float


def _std(values: list[float]) -> float:
    """Return sample standard deviation and keep zero for single value lists"""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_benchmark_cases(
    data_dir: Path, optima_file: str = "optima.json"
) -> list[BenchmarkCase]:
    """Load benchmark case metadata and known optima from JSON manifest"""
    manifest_path = data_dir / optima_file
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for name, item in payload.items():
        cases.append(
            BenchmarkCase(
                name=name,
                path=data_dir / item["file"],
                optimum=float(item["optimum"]),
            )
        )
    return cases


def _first_generation_at_or_below(history: Iterable[float], threshold: float) -> int | None:
    """Find first generation where best distance reaches the target threshold"""
    for generation, distance in enumerate(history):
        if distance <= threshold:
            return generation
    return None


def benchmark_instance(
    case: BenchmarkCase,
    base_config: ACOConfig,
    runs: int,
    target_gap: float,
    base_seed: int | None,
    start_city: int,
) -> InstanceStatistics:
    """Run repeated ACO solves for one instance and aggregate quality metrics"""
    instance = load_tsplib(case.path)
    threshold = case.optimum * (1.0 + target_gap)

    best_distances: list[float] = []
    relative_errors: list[float] = []
    best_generations: list[float] = []
    target_generations: list[float] = []
    runtimes: list[float] = []
    hits = 0

    for run_idx in range(runs):
        run_seed = None if base_seed is None else base_seed + run_idx
        config = replace(base_config, seed=run_seed)
        result = solve_tsp_aco(instance, config=config, start_city=start_city)

        best_distances.append(result.best_distance)
        relative_errors.append(((result.best_distance - case.optimum) / case.optimum) * 100.0)
        best_generations.append(float(result.best_generation))
        runtimes.append(result.runtime_seconds)

        target_generation = _first_generation_at_or_below(result.history_best, threshold)
        if target_generation is not None:
            hits += 1
            target_generations.append(float(target_generation))

    return InstanceStatistics(
        instance=case.name,
        optimum=case.optimum,
        runs=runs,
        best_distance_min=min(best_distances),
        best_distance_mean=statistics.mean(best_distances),
        best_distance_std=_std(best_distances),
        relative_error_mean_pct=statistics.mean(relative_errors),
        relative_error_std_pct=_std(relative_errors),
        accuracy_hit_rate_pct=(hits / runs) * 100.0,
        mean_best_generation=statistics.mean(best_generations),
        mean_target_generation=(
            statistics.mean(target_generations) if target_generations else None
        ),
        mean_runtime_seconds=statistics.mean(runtimes),
    )


def run_aco_benchmark(
    cases: list[BenchmarkCase],
    config: ACOConfig,
    runs: int,
    start_city: int = 0,
    target_gap: float = 0.05,
    base_seed: int | None = 1234,
) -> tuple[list[InstanceStatistics], OverallStatistics]:
    """Benchmark ACO across all configured cases and build overall summary stats"""
    summaries = [
        benchmark_instance(
            case=case,
            base_config=config,
            runs=runs,
            target_gap=target_gap,
            base_seed=base_seed,
            start_city=start_city,
        )
        for case in cases
    ]

    overall = OverallStatistics(
        runs_per_instance=runs,
        instances=len(summaries),
        mean_relative_error_pct=statistics.mean(
            summary.relative_error_mean_pct for summary in summaries
        ),
        mean_accuracy_hit_rate_pct=statistics.mean(
            summary.accuracy_hit_rate_pct for summary in summaries
        ),
        mean_runtime_seconds=statistics.mean(
            summary.mean_runtime_seconds for summary in summaries
        ),
    )
    return summaries, overall


def _print_report(
    summaries: list[InstanceStatistics], overall: OverallStatistics, target_gap: float
) -> None:
    """Print a compact table style report for per instance and overall metrics"""
    print(
        "instance  optimum  best_min  best_mean  gap_mean%  gap_std%  hit_rate%  "
        f"mean_best_gen  mean_hit_gen@{target_gap*100:.1f}%  mean_runtime_s"
    )
    for summary in summaries:
        mean_target = (
            f"{summary.mean_target_generation:.2f}"
            if summary.mean_target_generation is not None
            else "n/a"
        )
        print(
            f"{summary.instance:8} "
            f"{summary.optimum:7.1f} "
            f"{summary.best_distance_min:9.2f} "
            f"{summary.best_distance_mean:10.2f} "
            f"{summary.relative_error_mean_pct:9.2f} "
            f"{summary.relative_error_std_pct:8.2f} "
            f"{summary.accuracy_hit_rate_pct:8.2f} "
            f"{summary.mean_best_generation:13.2f} "
            f"{mean_target:22} "
            f"{summary.mean_runtime_seconds:14.4f}"
        )
    print(
        "\nOverall: "
        f"instances={overall.instances}, runs_per_instance={overall.runs_per_instance}, "
        f"mean_gap={overall.mean_relative_error_pct:.2f}%, "
        f"mean_hit_rate={overall.mean_accuracy_hit_rate_pct:.2f}%, "
        f"mean_runtime={overall.mean_runtime_seconds:.4f}s"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the ACO benchmark end to end"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/tsplib"))
    parser.add_argument("--optima-file", default="optima.json")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--target-gap", type=float, default=0.05)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--evaporation-rate", type=float, default=0.25)
    parser.add_argument("--pheromone-deposit", type=float, default=100.0)
    parser.add_argument("--elitist-weight", type=float, default=2.0)
    parser.add_argument("--min-pheromone", type=float, default=1e-6)
    parser.add_argument("--max-pheromone", type=float, default=1e6)
    parser.add_argument("--start-city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    cases = load_benchmark_cases(args.data_dir, args.optima_file)
    config = ACOConfig(
        population_size=args.population,
        generations=args.generations,
        alpha=args.alpha,
        beta=args.beta,
        evaporation_rate=args.evaporation_rate,
        pheromone_deposit=args.pheromone_deposit,
        elitist_weight=args.elitist_weight,
        min_pheromone=args.min_pheromone,
        max_pheromone=args.max_pheromone,
        seed=None,
    )
    summaries, overall = run_aco_benchmark(
        cases=cases,
        config=config,
        runs=args.runs,
        start_city=args.start_city,
        target_gap=args.target_gap,
        base_seed=args.seed,
    )

    _print_report(summaries, overall, args.target_gap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
