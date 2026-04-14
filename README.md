# QIAEA-for-TSP

This project solves the Travelling Salesman Problem (TSP) on TSPLIB instances using three metaheuristic approaches: a classical Genetic Algorithm (GA), Ant Colony Optimization (ACO), and a QIAEA variant inspired by quantum-inspired evolutionary ideas and adapted for routing problems. The repository includes standalone implementations of each algorithm, shared configuration through `core_params.py`, and a comparison script that runs all methods on the same instances and reports solution quality and runtime.

## How To Run

1. Configure parameters in `core_params.py`.
2. Run GA:
   - `py -m tsp_ga`
3. Run ACO:
   - `py -m tsp_aco`
4. Run QIAEA:
   - `py -m tsp_qiaea`
5. Run full comparison:
   - `py compare_algorithms.py`
