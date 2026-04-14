from math import inf
from pathlib import Path

# Shared dataset settings
DATA_DIR = Path("data/tsplib")
OPTIMA_FILE = "optima.json"

# Single-instance runs
GA_SINGLE_INSTANCE = DATA_DIR / "berlin52.tsp"
ACO_SINGLE_INSTANCE = DATA_DIR / "berlin52.tsp"
QIAEA_SINGLE_INSTANCE = DATA_DIR / "berlin52.tsp"
SINGLE_RUN_SEED = 42
START_CITY = 0

# Core GA parameters
GA_CORE_PARAMS = {
    "population_size": 220,
    "generations": 1000,
    "crossover_rate": 0.95,
    "mutation_rate": 0.20,
    "tournament_size": 7,
    "elitism": 4,
}

# Core ACO parameters
ACO_CORE_PARAMS = {
    "population_size": 20,
    "generations": 120,
    "alpha": 1.0,
    "beta": 3.0,
    "evaporation_rate": 0.25,
    "pheromone_deposit": 100.0,
    "elitist_weight": 2.0,
    "min_pheromone": 1e-6,
    "max_pheromone": 1e6,
}

# Core QIAEA parameters
QIAEA_CORE_PARAMS = {
    "population_size": 300,
    "generations": 1000,
    "n_males": 5,
    "elitism": 2,
    "substring_fraction": 0.10,
    "substring_fraction_end": 0.02,
    "bits_per_city": 16,
    "random_tie_break": True,
}

# Comparison settings (used by compare_algorithms.py)
COMPARISON_RUNS = 1
COMPARISON_BASE_SEED = 1234
COMPARISON_INSTANCE_NAMES: list[str] | None = None # None -> all from optima.json
COMPARISON_ENABLE_PLOTS = True
COMPARISON_PLOT_DIR = Path("comparison_plots")
COMPARISON_STRONG_ADVANTAGE_PCT = 5.0
COMPARISON_MAX_ADVANTAGE_PLOT_CASES = 40
COMPARISON_MAX_CASE_PLOTS = inf
