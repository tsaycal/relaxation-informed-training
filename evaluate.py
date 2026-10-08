"""
Evaluation pipeline: assess MILP formulation quality for trained models.

Measures:
  - LP relaxation gap (looseness of LP bound vs MIP optimum)
  - MILP solve time and node count
  - Number of unstable neurons
  - Prediction accuracy (MSE on test data)
  - Bound statistics (interval bound widths)
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from models import interval_bound_propagation_numpy, count_unstable_neurons
from formulations import solve_milp, solve_lp_relaxation, MILPResult


@dataclass
class EvalResult:
    """Comprehensive evaluation results."""
    # Prediction quality
    test_mse: float = 0.0
    test_mae: float = 0.0
    test_max_error: float = 0.0

    # MILP results (minimization)
    milp_min_obj: Optional[float] = None
    milp_min_time: float = 0.0
    milp_min_nodes: int = 0
    milp_min_gap: Optional[float] = None
    milp_min_x: Optional[list] = None

    # MILP results (maximization)
    milp_max_obj: Optional[float] = None
    milp_max_time: float = 0.0
    milp_max_nodes: int = 0
    milp_max_gap: Optional[float] = None
    milp_max_x: Optional[list] = None

    # LP relaxation
    lp_min_val: Optional[float] = None
    lp_max_val: Optional[float] = None
    lp_gap_min: Optional[float] = None  # MILP_min - LP_min (>= 0)
    lp_gap_max: Optional[float] = None  # LP_max - MILP_max (>= 0)

    # Bound statistics
    n_relu_neurons: int = 0
    n_unstable_neurons: int = 0
    n_binary_vars: int = 0
    mean_bound_width: float = 0.0
    max_bound_width: float = 0.0

    # Surrogate optimization quality (how close is MILP solution to true optimum)
    true_min_at_milp_sol: Optional[float] = None
    surrogate_gap: Optional[float] = None  # true f at MILP x* vs known global min

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    benchmark_func=None,
    known_minimum: Optional[float] = None,
    time_limit: float = 300.0,
    verbose: bool = True,
    y_min: float = 0.0,
    y_range: float = 1.0,
    normalized_mse: bool = True,
    solve_max: bool = True,
) -> EvalResult:
    """Full evaluation of a trained model.

    Args:
        model: trained ReLUNet
        X_test, y_test: test data
        input_lb, input_ub: input domain bounds
        benchmark_func: optional, the true function for surrogate quality
        known_minimum: optional, known global minimum of benchmark
        time_limit: MILP solve time limit
        verbose: print progress
        y_min: target normalization offset from training (TrainResult.y_min)
        y_range: target normalization scale from training (TrainResult.y_range)
        normalized_mse: if True (default), report MSE on the normalized [0,1]
            scale used during training; if False, denormalize predictions back
            to the original target scale before computing MSE.
        solve_max: if True (default), also solve the MILP and LP in the
            maximization direction.  Set to False to skip, saving roughly
            half the solve time when only the min direction is needed.

    Returns:
        EvalResult with all metrics
    """
    import torch

    result = EvalResult()

    # --- Prediction quality ---
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t).numpy().flatten()  # in normalized space

    if normalized_mse:
        # Compare on the [0, 1] scale consistent with training losses
        y_test_scaled = (y_test - y_min) / y_range
        result.test_mse = float(np.mean((preds - y_test_scaled) ** 2))
        result.test_mae = float(np.mean(np.abs(preds - y_test_scaled)))
        result.test_max_error = float(np.max(np.abs(preds - y_test_scaled)))
    else:
        # Denormalize predictions to the original target scale
        preds_unscaled = preds * y_range + y_min
        result.test_mse = float(np.mean((preds_unscaled - y_test) ** 2))
        result.test_mae = float(np.mean(np.abs(preds_unscaled - y_test)))
        result.test_max_error = float(np.max(np.abs(preds_unscaled - y_test)))

    if verbose:
        print(f"  Test MSE: {result.test_mse:.6f}, MAE: {result.test_mae:.6f}")

    # --- Bound statistics ---
    wb = model.get_weights_and_biases()
    layer_bounds = interval_bound_propagation_numpy(wb, input_lb, input_ub)

    result.n_relu_neurons = model.n_relu_neurons
    all_widths = []
    n_unstable = 0
    for l, (pre_lb, pre_ub, _, _) in enumerate(layer_bounds[:-1]):
        widths = pre_ub - pre_lb
        all_widths.extend(widths.tolist())
        n_unstable += int(np.sum((pre_lb < -1e-8) & (pre_ub > 1e-8)))

    result.n_unstable_neurons = n_unstable
    result.mean_bound_width = float(np.mean(all_widths)) if all_widths else 0.0
    result.max_bound_width = float(np.max(all_widths)) if all_widths else 0.0

    if verbose:
        print(
            f"  Neurons: {result.n_relu_neurons} total, "
            f"{result.n_unstable_neurons} unstable "
            f"({100 * result.n_unstable_neurons / max(result.n_relu_neurons, 1):.1f}%)"
        )
        print(f"  Bound widths: mean={result.mean_bound_width:.4f}, max={result.max_bound_width:.4f}")

    # --- MILP solve (minimize) ---
    if verbose:
        print("  Solving MILP (minimize)...")
    milp_min = solve_milp(wb, input_lb, input_ub, sense="minimize", time_limit=time_limit)
    result.milp_min_obj = milp_min.obj_val
    result.milp_min_time = milp_min.solve_time
    result.milp_min_nodes = milp_min.node_count
    result.milp_min_gap = milp_min.gap
    result.milp_min_x = milp_min.x_sol.tolist() if milp_min.x_sol is not None else None
    result.n_binary_vars = milp_min.n_binary_vars

    # --- MILP solve (maximize) ---
    if solve_max:
        if verbose:
            print("  Solving MILP (maximize)...")
        milp_max = solve_milp(wb, input_lb, input_ub, sense="maximize", time_limit=time_limit)
        result.milp_max_obj = milp_max.obj_val
        result.milp_max_time = milp_max.solve_time
        result.milp_max_nodes = milp_max.node_count
        result.milp_max_gap = milp_max.gap
        result.milp_max_x = milp_max.x_sol.tolist() if milp_max.x_sol is not None else None

    # --- LP relaxation ---
    if verbose:
        print("  Solving LP relaxations...")
    result.lp_min_val = solve_lp_relaxation(wb, input_lb, input_ub, sense="minimize")

    if solve_max:
        result.lp_max_val = solve_lp_relaxation(wb, input_lb, input_ub, sense="maximize")

    if result.milp_min_obj is not None and result.lp_min_val is not None:
        result.lp_gap_min = result.milp_min_obj - result.lp_min_val

    if solve_max and result.milp_max_obj is not None and result.lp_max_val is not None:
        result.lp_gap_max = result.lp_max_val - result.milp_max_obj

    if verbose:
        print(f"  LP gap (min): {result.lp_gap_min}")
        if solve_max:
            print(f"  LP gap (max): {result.lp_gap_max}")
        print(
            f"  MILP min: obj={result.milp_min_obj:.6f}, "
            f"time={result.milp_min_time:.2f}s, nodes={result.milp_min_nodes}"
        )

    # --- Surrogate optimization quality ---
    if benchmark_func is not None and milp_min.x_sol is not None:
        true_val = benchmark_func(milp_min.x_sol.reshape(1, -1))[0]
        result.true_min_at_milp_sol = float(true_val)
        if known_minimum is not None:
            result.surrogate_gap = float(true_val - known_minimum)
        if verbose:
            print(f"  True function at MILP min: {result.true_min_at_milp_sol:.6f}")
            if result.surrogate_gap is not None:
                print(f"  Surrogate gap (vs known min): {result.surrogate_gap:.6f}")

    return result
