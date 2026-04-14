import json
from pathlib import Path

from core_params import (
    DATA_DIR,
    OPTIMA_FILE,
    QIAEA_CORE_PARAMS,
    QIAEA_SINGLE_INSTANCE,
    SINGLE_RUN_SEED,
    START_CITY,
)
from .io import load_tsplib
from .qiaea import QIAEAConfig, solve_tsp_qiaea


def _find_reference_answer(instance_path: Path) -> float | None:
    """Find known optimum for the selected instance from optima manifest"""
    optima_path = DATA_DIR / OPTIMA_FILE
    if not optima_path.exists():
        return None

    payload = json.loads(optima_path.read_text(encoding="utf-8"))
    target_file = instance_path.name.lower()
    for item in payload.values():
        file_name = str(item.get("file", "")).replace("\\", "/")
        if Path(file_name).name.lower() == target_file:
            return float(item["optimum"])
    return None


def main() -> int:
    """Run one QIAEA solve with configured parameters and print a readable report"""
    instance_path = Path(QIAEA_SINGLE_INSTANCE)
    instance = load_tsplib(instance_path)
    config = QIAEAConfig(seed=SINGLE_RUN_SEED, **QIAEA_CORE_PARAMS)
    reference_answer = _find_reference_answer(instance_path)

    result = solve_tsp_qiaea(instance, config=config, start_city=START_CITY)
    route_1based = [city + 1 for city in result.best_route]

    print(f"Instance: {instance.name} ({instance.size} cities)")
    print(f"Input file: {instance_path}")
    print(
        "Algorithm: QIAEA | "
        f"population={config.population_size}, generations={config.generations}, "
        f"n_males={config.n_males}, elitism={config.elitism}, "
        f"substring_fraction={config.substring_fraction}, "
        f"substring_fraction_end={config.substring_fraction_end}, "
        f"bits_per_city={config.bits_per_city}, "
        f"random_tie_break={config.random_tie_break}, "
        f"start_city={START_CITY}, seed={config.seed}"
    )
    print(f"Best distance: {result.best_distance:.6f}")
    if reference_answer is None:
        print("Reference answer: not found in optima.json")
    else:
        gap_pct = ((result.best_distance - reference_answer) / reference_answer) * 100.0
        print(f"Reference answer: {reference_answer:.6f}")
        print(f"Gap vs answer: {gap_pct:+.2f}%")
    print("Best route (1-based): " + " -> ".join(map(str, route_1based + [route_1based[0]])))
    print(
        f"Convergence: start_best={result.history_best[0]:.6f}, "
        f"end_best={result.history_best[-1]:.6f}, "
        f"best_generation={result.best_generation}, runtime_s={result.runtime_seconds:.4f}"
    )
    return 0
