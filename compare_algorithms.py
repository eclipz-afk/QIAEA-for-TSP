import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from core_params import (
    ACO_CORE_PARAMS,
    COMPARISON_BASE_SEED,
    COMPARISON_ENABLE_PLOTS,
    COMPARISON_INSTANCE_NAMES,
    COMPARISON_MAX_CASE_PLOTS,
    COMPARISON_PLOT_DIR,
    COMPARISON_RUNS,
    COMPARISON_STRONG_ADVANTAGE_PCT,
    DATA_DIR,
    GA_CORE_PARAMS,
    OPTIMA_FILE,
    QIAEA_CORE_PARAMS,
    START_CITY,
)
from tsp_aco.aco import ACOConfig, solve_tsp_aco
from tsp_ga.ga import GAConfig, solve_tsp_ga
from tsp_ga.io import load_tsplib
from tsp_qiaea.qiaea import QIAEAConfig, solve_tsp_qiaea


@dataclass(frozen=True)
class Case:
    name: str
    path: Path
    optimum: float


@dataclass(frozen=True)
class CaseSummary:
    instance: str
    optimum: float
    runs: int
    ga_best_min: float
    ga_best_mean: float
    ga_best_std: float
    ga_gap_mean_pct: float
    ga_runtime_mean_s: float
    aco_best_min: float
    aco_best_mean: float
    aco_best_std: float
    aco_gap_mean_pct: float
    aco_runtime_mean_s: float
    qiaea_best_min: float
    qiaea_best_mean: float
    qiaea_best_std: float
    qiaea_gap_mean_pct: float
    qiaea_runtime_mean_s: float
    winner: str
    winner_advantage_pct: float
    mean_distance_delta: float


def _safe_pct(numerator: float, denominator: float) -> float:
    """Convert a ratio to percent and protect against division by zero"""
    if abs(denominator) < 1e-12:
        return 0.0
    return (numerator / denominator) * 100.0


def _parse_optional_int(value: str) -> int | None:
    """Parse optional integer CLI values and allow words like none or random"""
    normalized = value.strip().lower()
    if normalized in {"none", "null", "random"}:
        return None
    return int(value)


def _parse_instance_names(value: str) -> list[str] | None:
    """Parse a comma separated instance list or return None for all instances"""
    normalized = value.strip().lower()
    if normalized in {"all", "*", ""}:
        return None
    names = [name.strip() for name in value.split(",")]
    names = [name for name in names if name]
    return names or None


def _resolve_run_seed(base_seed: int | None, run_idx: int) -> int | None:
    """Build a deterministic per run seed from a base seed"""
    if base_seed is None:
        return None
    return base_seed + run_idx


def _parse_max_case_plots(value: str) -> float:
    """Parse max case plot limit and support inf for unlimited plotting"""
    normalized = value.strip().lower()
    if normalized in {"inf", "infinity"}:
        return math.inf
    parsed = int(value)
    if parsed < 0:
        raise ValueError("max-case-plots must be >= 0 or 'inf'.")
    return float(parsed)


def _load_cases(data_dir: Path, optima_file: str, selected: list[str] | None) -> list[Case]:
    """Load benchmark cases from the optima manifest and validate selected names"""
    payload = json.loads((data_dir / optima_file).read_text(encoding="utf-8"))
    available = {
        name: Case(
            name=name,
            path=data_dir / item["file"],
            optimum=float(item["optimum"]),
        )
        for name, item in payload.items()
    }
    if not selected:
        return list(available.values())

    missing = [name for name in selected if name not in available]
    if missing:
        preview = sorted(available)[:25]
        more_count = max(0, len(available) - len(preview))
        preview_text = ", ".join(preview)
        if more_count:
            preview_text += f", ... (+{more_count} more)"
        raise ValueError(
            f"Unknown instances in COMPARISON_INSTANCE_NAMES: {', '.join(missing)}. "
            f"Available: {preview_text}"
        )
    return [available[name] for name in selected]


def _assert_valid_route(route: tuple[int, ...], n_cities: int, *, algorithm: str) -> None:
    """Validate that a route is a full Hamiltonian cycle for the given instance size"""
    if len(route) == n_cities:
        core_route = route
    elif len(route) == n_cities + 1:
        if route[0] != route[-1]:
            raise RuntimeError(f"{algorithm} produced a non-closed cycle route.")
        core_route = route[:-1]
    else:
        raise RuntimeError(f"{algorithm} produced a route of unexpected length.")

    if len(set(core_route)) != n_cities:
        raise RuntimeError(f"{algorithm} produced a route with duplicate cities.")


def _assert_history_valid(history: tuple[float, ...], *, algorithm: str) -> None:
    """Check that convergence history is non empty and contains only finite values"""
    if not history:
        raise RuntimeError(f"{algorithm} returned an empty convergence history.")
    if not np.all(np.isfinite(np.array(history, dtype=np.float64))):
        raise RuntimeError(f"{algorithm} returned non-finite values in convergence history.")


def _trim_histories_to_common_length(
    ga_histories: np.ndarray,
    aco_histories: np.ndarray,
    qiaea_histories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim all history matrices to the same generation count for aligned plotting"""
    min_len = min(ga_histories.shape[1], aco_histories.shape[1], qiaea_histories.shape[1])
    return (
        ga_histories[:, :min_len],
        aco_histories[:, :min_len],
        qiaea_histories[:, :min_len],
    )


def _plot_case(
    case: Case,
    ga_histories: np.ndarray,
    aco_histories: np.ndarray,
    qiaea_histories: np.ndarray,
    ga_final: np.ndarray,
    aco_final: np.ndarray,
    qiaea_final: np.ndarray,
    ga_runtimes: np.ndarray,
    aco_runtimes: np.ndarray,
    qiaea_runtimes: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Create per instance quality and timing plots for all three algorithms"""
    ga_trim, aco_trim, qiaea_trim = _trim_histories_to_common_length(
        ga_histories=ga_histories,
        aco_histories=aco_histories,
        qiaea_histories=qiaea_histories,
    )
    ga_mean, ga_std = ga_trim.mean(axis=0), ga_trim.std(axis=0)
    aco_mean, aco_std = aco_trim.mean(axis=0), aco_trim.std(axis=0)
    q_mat_mean, q_mat_std = qiaea_trim.mean(axis=0), qiaea_trim.std(axis=0)

    x = np.arange(ga_mean.size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(x, ga_mean, label="GA", linewidth=2)
    axes[0].fill_between(x, ga_mean - ga_std, ga_mean + ga_std, alpha=0.2)
    axes[0].plot(x, aco_mean, label="ACO", linewidth=2)
    axes[0].fill_between(x, aco_mean - aco_std, aco_mean + aco_std, alpha=0.2)
    axes[0].plot(x, q_mat_mean, label="QIAEA", linewidth=2)
    axes[0].fill_between(x, q_mat_mean - q_mat_std, q_mat_mean + q_mat_std, alpha=0.2)
    axes[0].axhline(case.optimum, color="#d62728", linestyle="--", linewidth=1.8, label="Reference answer")
    axes[0].set_title(f"{case.name}: convergence")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best distance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].boxplot(
        [ga_final, aco_final, qiaea_final],
        tick_labels=["GA", "ACO", "QIAEA"],
        showmeans=True,
    )
    axes[1].set_title(f"{case.name}: final distance")
    axes[1].set_ylabel("Best distance")
    axes[1].axhline(case.optimum, color="#d62728", linestyle="--", linewidth=1.8, label="Reference answer")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case.name}_all_algorithms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    common_generations = ga_mean.size
    ga_time_per_gen = np.mean(ga_runtimes) / ga_histories.shape[1]
    aco_time_per_gen = np.mean(aco_runtimes) / aco_histories.shape[1]
    q_mat_time_per_gen = np.mean(qiaea_runtimes) / qiaea_histories.shape[1]

    ga_cumsum_time = np.arange(1, common_generations + 1) * ga_time_per_gen
    aco_cumsum_time = np.arange(1, common_generations + 1) * aco_time_per_gen
    q_mat_cumsum_time = np.arange(1, common_generations + 1) * q_mat_time_per_gen

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ga_cumsum_time, label="GA", linewidth=2)
    ax.plot(x, aco_cumsum_time, label="ACO", linewidth=2)
    ax.plot(x, q_mat_cumsum_time, label="QIAEA", linewidth=2)
    ax.set_title(f"{case.name}: cumulative time vs iteration")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    time_plot_path = output_dir / f"{case.name}_time_vs_iteration.png"
    fig.savefig(time_plot_path, dpi=150)
    plt.close(fig)

    return out_path, time_plot_path


def _plot_overall_averages(
    summaries: list[CaseSummary],
    output_dir: Path,
    strong_advantage_pct: float,
) -> Path:
    """Create an overall dashboard with average gaps runtimes and win counts"""
    ga_wins = sum(s.winner == "GA" for s in summaries)
    aco_wins = sum(s.winner == "ACO" for s in summaries)
    qiaea_wins = sum(s.winner == "QIAEA" for s in summaries)
    ties = len(summaries) - ga_wins - aco_wins - qiaea_wins

    ga_strong = sum(
        (s.winner == "GA") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )
    aco_strong = sum(
        (s.winner == "ACO") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )
    qiaea_strong = sum(
        (s.winner == "QIAEA") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(
        ["GA", "ACO", "QIAEA"],
        [
            statistics.mean(s.ga_gap_mean_pct for s in summaries),
            statistics.mean(s.aco_gap_mean_pct for s in summaries),
            statistics.mean(s.qiaea_gap_mean_pct for s in summaries),
        ],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    axes[0, 0].set_title("Average Gap To Optimum (%)")
    axes[0, 0].set_ylabel("Lower is better")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].bar(
        ["GA", "ACO", "QIAEA"],
        [
            statistics.mean(s.ga_runtime_mean_s for s in summaries),
            statistics.mean(s.aco_runtime_mean_s for s in summaries),
            statistics.mean(s.qiaea_runtime_mean_s for s in summaries),
        ],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    axes[0, 1].set_title("Average Runtime (seconds)")
    axes[0, 1].set_ylabel("Lower is better")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].bar(
        ["GA wins", "ACO wins", "QIAEA wins", "Ties"],
        [ga_wins, aco_wins, qiaea_wins, ties],
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"],
    )
    axes[1, 0].set_title("Case Wins By Mean Distance")
    axes[1, 0].set_ylabel("Number of instances")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].bar(
        ["GA", "ACO", "QIAEA"],
        [ga_strong, aco_strong, qiaea_strong],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    axes[1, 1].set_title(f"Strong Wins (>= {strong_advantage_pct:.1f}% better)")
    axes[1, 1].set_ylabel("Number of instances")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "overall_average_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _winner_and_advantage(
    ga_mean: float,
    aco_mean: float,
    qiaea_mean: float,
) -> tuple[str, float]:
    """Pick the winner by mean distance and compute advantage over second place"""
    means = {
        "GA": ga_mean,
        "ACO": aco_mean,
        "QIAEA": qiaea_mean,
    }

    winner = min(means, key=means.get)
    winner_mean = means[winner]
    second_best = min(value for name, value in means.items() if name != winner)
    if abs(winner_mean - second_best) <= 1e-12:
        return "TIE", 0.0
    return winner, _safe_pct(second_best - winner_mean, second_best)


def _summarize_case(
    case: Case,
    ga_final: np.ndarray,
    ga_runtime: np.ndarray,
    aco_final: np.ndarray,
    aco_runtime: np.ndarray,
    qiaea_final: np.ndarray,
    qiaea_runtime: np.ndarray,
    runs: int,
) -> CaseSummary:
    """Aggregate run arrays into a compact case summary record"""
    ga_mean = float(np.mean(ga_final))
    aco_mean = float(np.mean(aco_final))
    q_mat_mean = float(np.mean(qiaea_final))
    winner, winner_advantage_pct = _winner_and_advantage(
        ga_mean=ga_mean,
        aco_mean=aco_mean,
        qiaea_mean=q_mat_mean,
    )

    return CaseSummary(
        instance=case.name,
        optimum=case.optimum,
        runs=runs,
        ga_best_min=float(np.min(ga_final)),
        ga_best_mean=ga_mean,
        ga_best_std=float(np.std(ga_final, ddof=1) if runs > 1 else 0.0),
        ga_gap_mean_pct=((ga_mean - case.optimum) / case.optimum) * 100.0,
        ga_runtime_mean_s=float(np.mean(ga_runtime)),
        aco_best_min=float(np.min(aco_final)),
        aco_best_mean=aco_mean,
        aco_best_std=float(np.std(aco_final, ddof=1) if runs > 1 else 0.0),
        aco_gap_mean_pct=((aco_mean - case.optimum) / case.optimum) * 100.0,
        aco_runtime_mean_s=float(np.mean(aco_runtime)),
        qiaea_best_min=float(np.min(qiaea_final)),
        qiaea_best_mean=q_mat_mean,
        qiaea_best_std=float(np.std(qiaea_final, ddof=1) if runs > 1 else 0.0),
        qiaea_gap_mean_pct=((q_mat_mean - case.optimum) / case.optimum) * 100.0,
        qiaea_runtime_mean_s=float(np.mean(qiaea_runtime)),
        winner=winner,
        winner_advantage_pct=winner_advantage_pct,
        mean_distance_delta=ga_mean - aco_mean,
    )


def _print_case_summary(summary: CaseSummary) -> None:
    """Print a readable one block report for a single benchmark case"""
    print(
        f"[{summary.instance}] optimum={summary.optimum:.1f}, runs={summary.runs}\n"
        f"  GA    : best={summary.ga_best_min:.2f}, mean={summary.ga_best_mean:.2f}, "
        f"std={summary.ga_best_std:.2f}, gap_mean={summary.ga_gap_mean_pct:.2f}%, "
        f"runtime_mean={summary.ga_runtime_mean_s:.4f}s\n"
        f"  ACO   : best={summary.aco_best_min:.2f}, mean={summary.aco_best_mean:.2f}, "
        f"std={summary.aco_best_std:.2f}, gap_mean={summary.aco_gap_mean_pct:.2f}%, "
        f"runtime_mean={summary.aco_runtime_mean_s:.4f}s\n"
        f"  QIAEA : best={summary.qiaea_best_min:.2f}, mean={summary.qiaea_best_mean:.2f}, "
        f"std={summary.qiaea_best_std:.2f}, gap_mean={summary.qiaea_gap_mean_pct:.2f}%, "
        f"runtime_mean={summary.qiaea_runtime_mean_s:.4f}s\n"
        f"  Winner: {summary.winner} (mean-distance advantage={summary.winner_advantage_pct:.2f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    """Run a full comparison of GA ACO and QIAEA across selected TSPLIB cases"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=COMPARISON_RUNS)
    parser.add_argument("--base-seed", type=_parse_optional_int, default=COMPARISON_BASE_SEED)
    parser.add_argument(
        "--instances",
        type=_parse_instance_names,
        default=COMPARISON_INSTANCE_NAMES,
        help="Comma-separated instance names, or 'all'.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        dest="enable_plots",
        help="Enable comparison plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="enable_plots",
        help="Disable comparison plots.",
    )
    parser.add_argument(
        "--max-case-plots",
        type=_parse_max_case_plots,
        default=COMPARISON_MAX_CASE_PLOTS,
        help="Maximum number of instances that still produce per-case plots.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=COMPARISON_PLOT_DIR,
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--strong-advantage-pct",
        type=float,
        default=COMPARISON_STRONG_ADVANTAGE_PCT,
        help="Threshold used to count strong wins.",
    )
    parser.set_defaults(enable_plots=COMPARISON_ENABLE_PLOTS)
    args = parser.parse_args(argv)

    runs = max(1, int(args.runs))
    base_seed = args.base_seed
    selected_instances = args.instances
    enable_plots = bool(args.enable_plots)
    max_case_plots = float(args.max_case_plots)
    plot_dir = Path(args.plot_dir)
    strong_advantage_pct = float(args.strong_advantage_pct)

    cases = _load_cases(DATA_DIR, OPTIMA_FILE, selected_instances)
    ga_config = GAConfig(seed=None, **GA_CORE_PARAMS)
    aco_config = ACOConfig(seed=None, **ACO_CORE_PARAMS)
    qiaea_config = QIAEAConfig(seed=None, **QIAEA_CORE_PARAMS)

    print(
        "Comparison config: "
        f"cases={len(cases)}, runs={runs}, base_seed={base_seed}, "
        f"plots={enable_plots}, plot_dir={plot_dir}"
    )

    summaries: list[CaseSummary] = []
    generated_plots: list[Path] = []

    case_plotting_enabled = enable_plots and (len(cases) <= max_case_plots)
    if enable_plots and not case_plotting_enabled:
        print(
            "Per-instance convergence/box plots are skipped "
            f"because cases={len(cases)} > max_case_plots={max_case_plots}."
        )

    for case in cases:
        case_start = perf_counter()
        instance = load_tsplib(case.path)
        ga_runs = []
        aco_runs = []
        qiaea_runs = []
        print(
            f"\n[{case.name}] starting {runs} runs "
            f"(answer={case.optimum:.6f}, cities={instance.size})"
        )

        for run_idx in range(runs):
            run_seed = _resolve_run_seed(base_seed, run_idx)
            ga_result = solve_tsp_ga(instance, config=replace(ga_config, seed=run_seed))
            aco_result = solve_tsp_aco(
                instance,
                config=replace(aco_config, seed=run_seed),
                start_city=START_CITY,
            )
            qiaea_result = solve_tsp_qiaea(
                instance,
                config=replace(qiaea_config, seed=run_seed),
                start_city=START_CITY,
            )

            _assert_valid_route(ga_result.best_route, instance.size, algorithm="GA")
            _assert_valid_route(aco_result.best_route, instance.size, algorithm="ACO")
            _assert_valid_route(qiaea_result.best_route, instance.size, algorithm="QIAEA")
            _assert_history_valid(ga_result.history_best, algorithm="GA")
            _assert_history_valid(aco_result.history_best, algorithm="ACO")
            _assert_history_valid(qiaea_result.history_best, algorithm="QIAEA")

            ga_runs.append(ga_result)
            aco_runs.append(aco_result)
            qiaea_runs.append(qiaea_result)
            print(
                f"  run {run_idx + 1}/{runs} | seed={run_seed} | "
                f"GA={ga_result.best_distance:.2f} ({ga_result.runtime_seconds:.2f}s) | "
                f"ACO={aco_result.best_distance:.2f} ({aco_result.runtime_seconds:.2f}s) | "
                f"QIAEA={qiaea_result.best_distance:.2f} ({qiaea_result.runtime_seconds:.2f}s)"
            )

        ga_histories = np.array([run.history_best for run in ga_runs], dtype=np.float64)
        aco_histories = np.array([run.history_best for run in aco_runs], dtype=np.float64)
        qiaea_histories = np.array([run.history_best for run in qiaea_runs], dtype=np.float64)
        ga_final = np.array([run.best_distance for run in ga_runs], dtype=np.float64)
        aco_final = np.array([run.best_distance for run in aco_runs], dtype=np.float64)
        qiaea_final = np.array([run.best_distance for run in qiaea_runs], dtype=np.float64)
        ga_runtime = np.array([run.runtime_seconds for run in ga_runs], dtype=np.float64)
        aco_runtime = np.array([run.runtime_seconds for run in aco_runs], dtype=np.float64)
        qiaea_runtime = np.array([run.runtime_seconds for run in qiaea_runs], dtype=np.float64)

        summary = _summarize_case(
            case=case,
            ga_final=ga_final,
            ga_runtime=ga_runtime,
            aco_final=aco_final,
            aco_runtime=aco_runtime,
            qiaea_final=qiaea_final,
            qiaea_runtime=qiaea_runtime,
            runs=runs,
        )
        summaries.append(summary)
        _print_case_summary(summary)
        print(f"  case_elapsed_s={perf_counter() - case_start:.2f}")

        if case_plotting_enabled:
            plot_path, time_plot_path = _plot_case(
                case=case,
                ga_histories=ga_histories,
                aco_histories=aco_histories,
                qiaea_histories=qiaea_histories,
                ga_final=ga_final,
                aco_final=aco_final,
                qiaea_final=qiaea_final,
                ga_runtimes=ga_runtime,
                aco_runtimes=aco_runtime,
                qiaea_runtimes=qiaea_runtime,
                output_dir=plot_dir,
            )
            generated_plots.append(plot_path)
            generated_plots.append(time_plot_path)
            print(f"  plot  : {plot_path}")
            print(f"  time  : {time_plot_path}")

    if not summaries:
        print("No cases were evaluated.")
        return 0

    ga_mean_gap = statistics.mean(s.ga_gap_mean_pct for s in summaries)
    aco_mean_gap = statistics.mean(s.aco_gap_mean_pct for s in summaries)
    qiaea_mean_gap = statistics.mean(s.qiaea_gap_mean_pct for s in summaries)
    ga_mean_runtime = statistics.mean(s.ga_runtime_mean_s for s in summaries)
    aco_mean_runtime = statistics.mean(s.aco_runtime_mean_s for s in summaries)
    qiaea_mean_runtime = statistics.mean(s.qiaea_runtime_mean_s for s in summaries)

    ga_wins = sum(s.winner == "GA" for s in summaries)
    aco_wins = sum(s.winner == "ACO" for s in summaries)
    qiaea_wins = sum(s.winner == "QIAEA" for s in summaries)
    ties = len(summaries) - ga_wins - aco_wins - qiaea_wins

    ga_strong = sum(
        (s.winner == "GA") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )
    aco_strong = sum(
        (s.winner == "ACO") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )
    qiaea_strong = sum(
        (s.winner == "QIAEA") and (s.winner_advantage_pct >= strong_advantage_pct)
        for s in summaries
    )

    print("\nOverall averages:")
    print(f"  GA    mean_gap={ga_mean_gap:.2f}% | mean_runtime={ga_mean_runtime:.4f}s")
    print(f"  ACO   mean_gap={aco_mean_gap:.2f}% | mean_runtime={aco_mean_runtime:.4f}s")
    print(f"  QIAEA mean_gap={qiaea_mean_gap:.2f}% | mean_runtime={qiaea_mean_runtime:.4f}s")

    print("\nWin counts by mean best distance:")
    print(f"  GA={ga_wins}, ACO={aco_wins}, QIAEA={qiaea_wins}, ties={ties}")
    print(
        f"Strong wins (>= {strong_advantage_pct:.1f}% better): "
        f"GA={ga_strong}, ACO={aco_strong}, QIAEA={qiaea_strong}"
    )

    # Keep only per-instance plots:
    # - <instance>_all_algorithms.png
    # - <instance>_time_vs_iteration.png

    if generated_plots:
        print("\nSaved plots:")
        for path in generated_plots:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
