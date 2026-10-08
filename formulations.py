"""
MILP formulations for ReLU neural networks using Gurobi.

Implements the big-M formulation for embedding a trained ReLU network
into a mixed-integer linear program. Bounds are computed via interval
arithmetic (IBP) and used as big-M constants.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from models import interval_bound_propagation_numpy


@dataclass
class MILPResult:
    """Container for MILP solution results."""
    status: int
    obj_val: Optional[float] = None
    obj_bound: Optional[float] = None
    lp_relaxation_val: Optional[float] = None
    gap: Optional[float] = None          # MIP gap reported by Gurobi
    lp_gap: Optional[float] = None       # gap between LP relaxation and MIP optimum
    solve_time: float = 0.0
    node_count: int = 0
    x_sol: Optional[np.ndarray] = None
    n_binary_vars: int = 0
    n_unstable_neurons: int = 0
    # Per-sample LP relaxation gaps (for detailed analysis)
    extra: Dict[str, Any] = field(default_factory=dict)


def build_bigm_milp(
    weights_and_biases: list,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    sense: str = "minimize",
    output_index: int = 0,
    env: Optional[gp.Env] = None,
) -> Tuple[gp.Model, dict]:
    """Build a big-M MILP formulation of a ReLU network.

    The MILP encodes y = NN(x) as linear constraints with binary variables
    for each ReLU neuron, then optimizes y over the input domain.

    For each hidden neuron j in layer l with pre-activation z_j = W_j @ x_prev + b_j:
        x_j >= z_j              (active: x_j = z_j)
        x_j >= 0                (always non-negative)
        x_j <= z_j - L_j*(1-a_j)   (if inactive, allows x_j = 0)
        x_j <= U_j * a_j           (if inactive, forces x_j = 0)

    where L_j, U_j are pre-activation bounds and a_j is binary.

    Args:
        weights_and_biases: list of (W, b) numpy arrays
        input_lb: shape (input_dim,), lower bounds on input
        input_ub: shape (input_dim,), upper bounds on input
        sense: 'minimize' or 'maximize'
        output_index: which output to optimize (for multi-output networks)
        env: optional Gurobi environment

    Returns:
        (model, var_dict) where var_dict contains variable references
    """
    # Compute interval bounds
    layer_bounds = interval_bound_propagation_numpy(weights_and_biases, input_lb, input_ub)
    n_layers = len(weights_and_biases)
    input_dim = len(input_lb)

    # Create model
    if env is not None:
        m = gp.Model("relu_bigm", env=env)
    else:
        m = gp.Model("relu_bigm")
    m.Params.OutputFlag = 0

    # Input variables
    x_in = m.addMVar(input_dim, lb=input_lb, ub=input_ub, name="x_in")

    # Build layer by layer
    var_dict = {"x_in": x_in, "layers": []}
    prev_x = x_in

    n_unstable = 0

    for l, (W, b) in enumerate(weights_and_biases):
        is_output = l == n_layers - 1
        n_neurons = W.shape[0]
        pre_lb, pre_ub, post_lb, post_ub = layer_bounds[l]

        # Pre-activation: z = W @ prev_x + b
        z = m.addMVar(n_neurons, lb=pre_lb, ub=pre_ub, name=f"z_{l}")
        m.addConstr(z == W @ prev_x + b, name=f"pre_act_{l}")

        if is_output:
            var_dict["layers"].append({"z": z})
            var_dict["output"] = z
        else:
            # Post-ReLU variables
            x = m.addMVar(n_neurons, lb=post_lb, ub=post_ub, name=f"x_{l}")

            # Binary activation variables (only for unstable neurons)
            layer_info = {"z": z, "x": x, "binary_indices": [], "a": None}

            unstable_mask = (pre_lb < -1e-8) & (pre_ub > 1e-8)
            stable_active = pre_lb >= -1e-8   # always active
            stable_inactive = pre_ub <= 1e-8  # always inactive

            # Handle stable neurons (no binary variable needed)
            for j in range(n_neurons):
                if stable_active[j]:
                    # x_j = z_j (always active)
                    m.addConstr(x[j] == z[j], name=f"stable_active_{l}_{j}")
                elif stable_inactive[j]:
                    # x_j = 0 (always inactive)
                    m.addConstr(x[j] == 0, name=f"stable_inactive_{l}_{j}")

            # Handle unstable neurons with big-M
            unstable_indices = np.where(unstable_mask)[0]
            n_unstable += len(unstable_indices)

            if len(unstable_indices) > 0:
                a = m.addMVar(
                    len(unstable_indices), vtype=GRB.BINARY,
                    name=f"a_{l}",
                )
                layer_info["a"] = a
                layer_info["binary_indices"] = unstable_indices.tolist()

                for idx_a, j in enumerate(unstable_indices):
                    L_j = pre_lb[j]  # negative
                    U_j = pre_ub[j]  # positive

                    # x_j >= z_j
                    m.addConstr(x[j] >= z[j], name=f"relu_lb1_{l}_{j}")
                    # x_j >= 0 (handled by variable bound)

                    # x_j <= z_j - L_j * (1 - a_j)
                    m.addConstr(
                        x[j] <= z[j] - L_j * (1 - a[idx_a]),
                        name=f"relu_ub1_{l}_{j}",
                    )
                    # x_j <= U_j * a_j
                    m.addConstr(
                        x[j] <= U_j * a[idx_a],
                        name=f"relu_ub2_{l}_{j}",
                    )

            var_dict["layers"].append(layer_info)
            prev_x = x

    # Set objective
    obj_var = var_dict["output"]
    if sense == "minimize":
        m.setObjective(obj_var[output_index], GRB.MINIMIZE)
    else:
        m.setObjective(obj_var[output_index], GRB.MAXIMIZE)

    m.update()
    var_dict["n_unstable"] = n_unstable

    return m, var_dict


def solve_milp(
    weights_and_biases: list,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    sense: str = "minimize",
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    env: Optional[gp.Env] = None,
) -> MILPResult:
    """Build and solve the big-M MILP for a ReLU network.

    Args:
        weights_and_biases: list of (W, b) numpy arrays
        input_lb, input_ub: input domain bounds
        sense: 'minimize' or 'maximize'
        time_limit: Gurobi time limit in seconds
        mip_gap: Gurobi MIP gap tolerance
        env: optional Gurobi environment

    Returns:
        MILPResult with solution details
    """
    m, var_dict = build_bigm_milp(
        weights_and_biases, input_lb, input_ub, sense=sense, env=env,
    )
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap

    m.optimize()

    result = MILPResult(
        status=m.Status,
        solve_time=m.Runtime,
        n_binary_vars=sum(1 for v in m.getVars() if v.VType == GRB.BINARY),
        n_unstable_neurons=var_dict["n_unstable"],
    )

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        if m.SolCount > 0:
            result.obj_val = m.ObjVal
            result.obj_bound = m.ObjBound
            result.gap = m.MIPGap
            result.node_count = int(m.NodeCount)
            result.x_sol = np.array([v.X for v in var_dict["x_in"].tolist()])

    return result


def solve_lp_relaxation(
    weights_and_biases: list,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    sense: str = "minimize",
    env: Optional[gp.Env] = None,
) -> Optional[float]:
    """Solve the LP relaxation of the big-M MILP.

    Args:
        weights_and_biases: list of (W, b) numpy arrays
        input_lb, input_ub: input domain bounds
        sense: 'minimize' or 'maximize'
        env: optional Gurobi environment

    Returns:
        LP relaxation optimal value, or None if infeasible
    """
    m, var_dict = build_bigm_milp(
        weights_and_biases, input_lb, input_ub, sense=sense, env=env,
    )

    # Relax binary variables to continuous [0, 1]
    m = m.relax()
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return m.ObjVal
    return None


def compute_lp_gap(
    weights_and_biases: list,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    time_limit: float = 1800.0,
    env: Optional[gp.Env] = None,
) -> Dict[str, Any]:
    """Compute the LP relaxation gap for both min and max directions.

    The LP gap measures how loose the LP relaxation is compared to the
    true MILP optimum. Smaller gap = tighter relaxation.

    Args:
        weights_and_biases: list of (W, b) numpy arrays
        input_lb, input_ub: input domain bounds
        time_limit: Gurobi time limit
        env: optional Gurobi environment

    Returns:
        Dictionary with LP and MILP results for both min/max
    """
    results = {}

    for sense in ["minimize", "maximize"]:
        milp_result = solve_milp(
            weights_and_biases, input_lb, input_ub,
            sense=sense, time_limit=time_limit, env=env,
        )
        lp_val = solve_lp_relaxation(
            weights_and_biases, input_lb, input_ub,
            sense=sense, env=env,
        )

        results[sense] = {
            "milp": milp_result,
            "lp_val": lp_val,
        }

        if milp_result.obj_val is not None and lp_val is not None:
            if sense == "minimize":
                # LP relaxation gives a lower bound for minimization
                results[sense]["lp_gap"] = milp_result.obj_val - lp_val
            else:
                # LP relaxation gives an upper bound for maximization
                results[sense]["lp_gap"] = lp_val - milp_result.obj_val
        else:
            results[sense]["lp_gap"] = None

    return results
