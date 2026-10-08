#!/usr/bin/env python3
"""
Run a single benchmark experiment: train one ReLU network with one
regularizer setting and one random seed, then evaluate the resulting
MILP formulation (unstable neurons, LP relaxation gap, MILP solve time).

The paper reports means over 20 seeds for every combination of
benchmark, architecture, regularizer, and regularization weight. Each
combination is an independent run of this script, so the full grid can
be parallelized in whatever way suits your environment (e.g., a job
array on a cluster).

Usage:
    python run_experiment.py --benchmark peaks --depth 3 --regularizer bound_width --lam 1e-3 --seed 0
    python run_experiment.py --benchmark ackley --depth 5 --regularizer bw+lp --lam 1e-3

Options:
    --benchmark    peaks | himmelblau | ackley   (all 2-dimensional)
    --depth        number of hidden layers (2, 3, or 5 in the paper), width 25
    --regularizer  none | l1 | l2 | bound_width | stable_neuron | lp_gap | bw+lp
    --lam          regularization weight lambda (paper: 1e-4, 1e-3, 1e-2)
    --seed         random seed (paper: 0, ..., 19)
"""

import os
import json
import argparse
import torch

from benchmarks import get_benchmark, generate_data
from models import create_model
from regularizers import (
    get_regularizer,
    CombinedRegularizer,
    BoundWidthRegularizer,
    LPRelaxationGapRegularizer,
)
from train import train_model, TrainConfig
from evaluate import evaluate_model


# ============================================================
# Settings used in the paper
# ============================================================

DIM = 2                  # all benchmarks are 2-dimensional
WIDTH = 25               # neurons per hidden layer
N_TRAIN = 70_000         # training samples (Latin hypercube)
N_VAL = 30_000           # validation samples
N_TEST = 10_000          # held-out test samples
TRAIN_EPOCHS = 200
LP_SAMPLES = 8           # |B_s|: LP solves per mini-batch for R_LP
MILP_TIME_LIMIT = 1800   # seconds


def build_regularizer(name):
    """Construct a regularizer by name.

    For the combined regularizer, lambda multiplies the sum
    R_BW + R_LP (i.e., alpha = 1 in Proposition 3).
    """
    if name == "lp_gap":
        return LPRelaxationGapRegularizer(sense="min", n_samples=LP_SAMPLES)
    if name == "bw+lp":
        return CombinedRegularizer([
            (BoundWidthRegularizer(), 1.0),
            (LPRelaxationGapRegularizer(sense="min", n_samples=LP_SAMPLES), 1.0),
        ])
    return get_regularizer(name)


def run_single_experiment(benchmark_name, depth, reg_name, reg_weight,
                          seed=0, output_root="results", verbose=True):
    """Train and evaluate one (benchmark, architecture, regularizer, seed)."""
    hidden_dims = [WIDTH] * depth
    arch_str = "-".join(str(d) for d in [DIM] + hidden_dims + [1])
    if reg_name == "none":
        reg_weight = 0.0

    if verbose:
        print(f"{benchmark_name} | {arch_str} | {reg_name} "
              f"(lambda={reg_weight}) | seed={seed}")

    benchmark = get_benchmark(benchmark_name, dim=DIM)
    input_lb = benchmark.domain[:, 0]
    input_ub = benchmark.domain[:, 1]

    # Data
    X_train, y_train = generate_data(benchmark, N_TRAIN, method="lhs", seed=seed)
    X_val, y_val = generate_data(benchmark, N_VAL, method="lhs", seed=seed + 1)
    X_test, y_test = generate_data(benchmark, N_TEST, method="lhs", seed=seed + 2)

    # Train (the parameters with the lowest validation loss are retained)
    model = create_model(DIM, hidden_dims, output_dim=1, seed=seed)
    train_config = TrainConfig(epochs=TRAIN_EPOCHS, seed=seed, reg_weight=reg_weight)
    train_result = train_model(
        model, X_train, y_train, X_val, y_val,
        input_lb, input_ub, build_regularizer(reg_name), train_config,
        verbose=verbose,
    )

    # Evaluate the downstream MILP (minimization)
    eval_result = evaluate_model(
        model, X_test, y_test, input_lb, input_ub,
        benchmark_func=benchmark.func,
        known_minimum=benchmark.known_minimum,
        time_limit=MILP_TIME_LIMIT,
        solve_max=False,
        verbose=verbose,
        y_min=train_result.y_min,
        y_range=train_result.y_range,
    )

    # Save
    tag = reg_name if reg_name == "none" else f"{reg_name}_{reg_weight:g}"
    output_dir = os.path.join(output_root, benchmark_name, arch_str, tag, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    experiment = {
        "benchmark": benchmark_name,
        "architecture": arch_str,
        "regularizer": reg_name,
        "reg_weight": reg_weight,
        "seed": seed,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "epochs": TRAIN_EPOCHS,
        "train": {
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
        },
        "eval": eval_result.to_dict(),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    if verbose:
        print(f"Results saved to {output_dir}")
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate one regularized ReLU surrogate model."
    )
    parser.add_argument("--benchmark", default="peaks",
                        choices=["peaks", "himmelblau", "ackley"])
    parser.add_argument("--depth", type=int, default=2,
                        help="Number of hidden layers (width 25 each)")
    parser.add_argument("--regularizer", default="none",
                        choices=["none", "l1", "l2", "bound_width",
                                 "stable_neuron", "lp_gap", "bw+lp"])
    parser.add_argument("--lam", type=float, default=1e-3,
                        help="Regularization weight lambda (ignored for 'none')")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results",
                        help="Root directory for results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_single_experiment(
        args.benchmark, args.depth, args.regularizer, args.lam,
        seed=args.seed, output_root=args.output, verbose=not args.quiet,
    )
