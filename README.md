# Relaxation-Informed Training of Neural Network Surrogate Models

Code for the benchmark-function experiments (direct optimization over ReLU network surrogates) in:

> Tsay (2026). *Relaxation-Informed Training of Neural Network Surrogate Models.*

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a working installation of [Gurobi](https://www.gurobi.com/) (free academic licences are available). LPs inside the training loop are solved with HiGHS via `scipy.optimize`.

## Running experiments

Each call to `run_experiment.py` trains one network (one benchmark, architecture, regularizer, regularization weight, and random seed) and then evaluates the downstream MILP:

```bash
python run_experiment.py --benchmark peaks --depth 3 --regularizer bound_width --lam 1e-3 --seed 0
```

| Option | Values |
|--------|--------|
| `--benchmark` | `peaks`, `himmelblau`, `ackley` (2-dimensional) |
| `--depth` | number of hidden layers, each with 25 neurons (2, 3, or 5 in the paper) |
| `--regularizer` | `none`, `l1`, `l2`, `bound_width`, `stable_neuron`, `lp_gap`, `bw+lp` |
| `--lam` | regularization weight λ (1e-4, 1e-3, or 1e-2 in the paper) |
| `--seed` | random seed (0–19 in the paper) |

The paper reports means over 20 seeds for each combination. Runs are independent, so the full grid can be parallelized however suits your environment (e.g., a cluster job array). Note that training with 70,000 samples for 200 epochs, and especially the LP-based regularizers, can take a substantial amount of time per run.

Results (a `results.json` with training and MILP metrics, plus the trained `model.pt`) are written to `results/<benchmark>/<architecture>/<regularizer>/seed_<seed>/`.

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Runs one training + evaluation experiment |
| `models.py` | ReLU network definition and interval bound propagation (IBP) |
| `train.py` | Training loop with regularization |
| `regularizers.py` | Regularizers: L1, L2, bound width, stable neuron, LP relaxation gap |
| `formulations.py` | Big-M MILP and LP relaxation formulations (Gurobi) |
| `evaluate.py` | MILP/LP evaluation of trained models |
| `benchmarks.py` | Peaks, Himmelblau, and Ackley test functions and data generation |
