"""
Regularization terms for training ReLU networks with tight MILP relaxations.

Each regularizer is a callable that takes (model, input_lb, input_ub) and
returns a scalar loss term (differentiable w.r.t. model parameters).

Regularizers that need per-sample information (LP gap) additionally accept
an optional X_batch argument. The train.py loop always passes X_batch; the
bound-based regularizers simply ignore it.

LP relaxation gap regularizer
------------------------------
For each training sample x_i, the LP relaxation gap is:
    gap_max(x_i) = LP_max(x_i) - f_theta(x_i)  >= 0
    gap_min(x_i) = f_theta(x_i) - LP_min(x_i)  >= 0

Gradient computation uses the envelope theorem: at the LP optimum, the dual
variables (shadow prices) for the equality constraints z^l = W^l x^{l-1} + b^l
give the sensitivity of the LP value to the network parameters. We implement
this via a "straight-through" proxy that has the correct gradient but whose
forward value is replaced with the true LP solution.

Note: big-M values (from IBP) are treated as constants. The LP gap regularizer
captures weight->constraint-RHS sensitivity, while BoundWidthRegularizer
captures weight->big-M sensitivity. The two are complementary.
"""

import numpy as np
import torch
from scipy.optimize import linprog
from typing import List, Tuple, Optional

from models import interval_bound_propagation, interval_bound_propagation_numpy


# ========================================================================== #
#  Base regularizers                                                          #
# ========================================================================== #

class BaseRegularizer:
    """Base class for regularizers."""

    name: str = "none"

    def __call__(
        self,
        model: "ReLUNet",
        input_lb: torch.Tensor,
        input_ub: torch.Tensor,
        X_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the regularization term.

        Args:
            model: ReLUNet instance
            input_lb: shape (input_dim,), lower bounds on inputs
            input_ub: shape (input_dim,), upper bounds on inputs
            X_batch: shape (batch_size, input_dim), current mini-batch inputs.
                     Required by LP-based regularizers; ignored by others.

        Returns:
            Scalar tensor (the regularization loss)
        """
        return torch.tensor(0.0)


class L1Regularizer(BaseRegularizer):
    """Standard L1 weight regularization."""

    name = "l1"

    def __call__(self, model, input_lb, input_ub, X_batch=None):
        return sum(p.abs().sum() for p in model.parameters())


class L2Regularizer(BaseRegularizer):
    """Standard L2 weight regularization (weight decay)."""

    name = "l2"

    def __call__(self, model, input_lb, input_ub, X_batch=None):
        return sum((p**2).sum() for p in model.parameters())


class BoundWidthRegularizer(BaseRegularizer):
    """Penalizes the width of interval arithmetic bounds.

    Encourages the network to have tight pre-activation bounds,
    which directly translates to smaller big-M values in the MILP.

    Loss = mean of (pre_ub - pre_lb) over all hidden neurons.
    """

    name = "bound_width"

    def __call__(self, model, input_lb, input_ub, X_batch=None):
        layer_bounds = interval_bound_propagation(model, input_lb, input_ub)

        widths = torch.cat([
            (pre_ub - pre_lb).flatten()
            for pre_lb, pre_ub, _, _ in layer_bounds[:-1]
        ])
        return widths.mean()


class StableNeuronRegularizer(BaseRegularizer):
    """Penalizes unstable neurons (pre-activation bounds straddling zero).

    For each neuron where pre_lb < 0 < pre_ub, the penalty is
    min(-pre_lb, pre_ub), which encourages either the lower bound
    to rise above zero or the upper bound to fall below zero.
    Stable neurons contribute 0.

    Loss = mean of min(-pre_lb, pre_ub) over all hidden neurons.
    """

    name = "stable_neuron"

    def __call__(self, model, input_lb, input_ub, X_batch=None):
        layer_bounds = interval_bound_propagation(model, input_lb, input_ub)

        penalties = torch.cat([
            torch.minimum(
                torch.clamp(-pre_lb, min=0),
                torch.clamp(pre_ub,  min=0),
            ).flatten()
            for pre_lb, pre_ub, _, _ in layer_bounds[:-1]
        ])
        return penalties.mean()


class CombinedRegularizer(BaseRegularizer):
    """Combines multiple regularizers with individual weights.

    Example:
        reg = CombinedRegularizer([
            (L2Regularizer(), 1e-4),
            (BoundWidthRegularizer(), 1e-3),
        ])
    """

    name = "combined"

    def __init__(self, regularizers_and_weights: list):
        self.regularizers_and_weights = regularizers_and_weights
        names = [r.name for r, _ in regularizers_and_weights]
        self.name = "+".join(names)

    def __call__(self, model, input_lb, input_ub, X_batch=None):
        total = torch.tensor(0.0)
        for reg, weight in self.regularizers_and_weights:
            total = total + weight * reg(model, input_lb, input_ub, X_batch=X_batch)
        return total


# ========================================================================== #
#  LP relaxation gap regularizer                                              #
# ========================================================================== #

def _build_and_solve_lp(
    wb_numpy: list,
    layer_bounds: list,
    x_i: np.ndarray,
    sense: str,
) -> Tuple[Optional[float], Optional[list], Optional[list]]:
    """Build and solve the LP relaxation with input fixed to x_i.

    Returns
    -------
    lp_value : float or None
        The LP optimal value (None if infeasible / solver error).
    dual_per_layer : list of np.ndarray or None
        Dual variables nu^l for the equality constraints z^l = W^l x^{l-1} + b^l.
        Signed so that d(lp_value)/d(b^l) = nu^l.
    lp_x_primals : list of np.ndarray or None
        LP primal values [x_i, x^1*, ..., x^{L-1}*] for post-activation
        variables (used as constants in gradient formula).
    """
    n_layers = len(wb_numpy)

    # ---- Variable layout ------------------------------------------------
    var_offset = {}
    n_vars = 0
    unstable_per_layer = []
    eq_row_start = {}
    n_eq_total = 0

    for l, (W, b) in enumerate(wb_numpy):
        n_l = W.shape[0]
        is_out = (l == n_layers - 1)
        pre_lb, pre_ub, _, _ = layer_bounds[l]

        var_offset[('z', l)] = n_vars;  n_vars += n_l
        if not is_out:
            var_offset[('x', l)] = n_vars;  n_vars += n_l
            unstable = np.where((pre_lb < -1e-8) & (pre_ub > 1e-8))[0]
            unstable_per_layer.append(unstable)
            var_offset[('a', l)] = n_vars;  n_vars += len(unstable)
        else:
            unstable_per_layer.append(np.array([], dtype=int))

        eq_row_start[l] = n_eq_total
        n_eq_total += n_l

    # ---- Objective ------------------------------------------------------
    c = np.zeros(n_vars)
    out_off = var_offset[('z', n_layers - 1)]
    c[out_off] = -1.0 if sense == "max" else 1.0

    # ---- Equality constraints: z^l = W^l @ x^{l-1} + b^l ---------------
    A_eq = np.zeros((n_eq_total, n_vars))
    b_eq = np.zeros(n_eq_total)

    for l, (W, b) in enumerate(wb_numpy):
        n_l = W.shape[0]
        rs = eq_row_start[l]
        re = rs + n_l
        z_off = var_offset[('z', l)]

        A_eq[rs:re, z_off:z_off + n_l] = np.eye(n_l)

        if l == 0:
            b_eq[rs:re] = W @ x_i + b
        else:
            x_prev_off = var_offset[('x', l - 1)]
            n_prev = wb_numpy[l - 1][0].shape[0]
            A_eq[rs:re, x_prev_off:x_prev_off + n_prev] = -W
            b_eq[rs:re] = b

    # ---- Inequality constraints (big-M for unstable neurons) ------------
    A_ub_list, b_ub_list = [], []

    for l in range(n_layers - 1):
        pre_lb, pre_ub, _, _ = layer_bounds[l]
        n_l = wb_numpy[l][0].shape[0]
        z_off = var_offset[('z', l)]
        x_off = var_offset[('x', l)]
        a_off = var_offset[('a', l)]
        unstable = unstable_per_layer[l]
        stable_act = np.where(pre_lb >= -1e-8)[0]

        for idx_a, j in enumerate(unstable):
            L_j, U_j = pre_lb[j], pre_ub[j]

            r = np.zeros(n_vars)
            r[z_off + j] = 1.0;  r[x_off + j] = -1.0
            A_ub_list.append(r);  b_ub_list.append(0.0)

            r = np.zeros(n_vars)
            r[x_off + j] = 1.0;  r[z_off + j] = -1.0;  r[a_off + idx_a] = -L_j
            A_ub_list.append(r);  b_ub_list.append(-L_j)

            r = np.zeros(n_vars)
            r[x_off + j] = 1.0;  r[a_off + idx_a] = -U_j
            A_ub_list.append(r);  b_ub_list.append(0.0)

        for j in stable_act:
            r = np.zeros(n_vars)
            r[x_off + j] = 1.0;  r[z_off + j] = -1.0
            A_ub_list.append(r);  b_ub_list.append(0.0)
            r = np.zeros(n_vars)
            r[x_off + j] = -1.0;  r[z_off + j] = 1.0
            A_ub_list.append(r);  b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub_arr = np.array(b_ub_list) if b_ub_list else None

    # ---- Variable bounds ------------------------------------------------
    bounds = []
    for l, (W, b) in enumerate(wb_numpy):
        n_l = W.shape[0]
        is_out = (l == n_layers - 1)
        pre_lb, pre_ub, post_lb, post_ub = layer_bounds[l]

        for j in range(n_l):
            bounds.append((pre_lb[j], pre_ub[j]))

        if not is_out:
            for j in range(n_l):
                bounds.append((post_lb[j], post_ub[j]))
            for _ in unstable_per_layer[l]:
                bounds.append((0.0, 1.0))

    # ---- Solve ----------------------------------------------------------
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub_arr,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method='highs',
    )

    if res.status != 0:
        return None, None, None

    # ---- Extract results ------------------------------------------------
    lp_value = float(-res.fun if sense == "max" else res.fun)

    raw_marginals = res.eqlin.marginals
    sign = -1.0 if sense == "max" else 1.0
    dual_flat = sign * raw_marginals

    dual_per_layer = []
    for l in range(n_layers):
        rs = eq_row_start[l]
        n_l = wb_numpy[l][0].shape[0]
        dual_per_layer.append(dual_flat[rs: rs + n_l])

    lp_x_primals = [x_i]
    for l in range(n_layers - 1):
        x_off = var_offset[('x', l)]
        n_l = wb_numpy[l][0].shape[0]
        lp_x_primals.append(res.x[x_off: x_off + n_l])

    return lp_value, dual_per_layer, lp_x_primals


def _lp_gap_differentiable(
    model: "ReLUNet",
    x_i: torch.Tensor,
    layer_bounds: list,
    wb_numpy: list,
    f_theta_val: torch.Tensor,
    sense: str,
) -> Optional[torch.Tensor]:
    """Compute a differentiable LP gap tensor for one sample.

    The returned tensor has:
      - VALUE  = LP_val - f_theta(x_i)   (for sense='max')
               = f_theta(x_i) - LP_val   (for sense='min')
      - GRADIENT w.r.t. model parameters: via envelope theorem duals.
    """
    x_i_np = x_i.detach().cpu().numpy()

    lp_value, dual_per_layer, lp_x_primals = _build_and_solve_lp(
        wb_numpy, layer_bounds, x_i_np, sense,
    )
    if lp_value is None:
        return None

    proxy = torch.zeros(1)
    for l, layer in enumerate(model.layers):
        nu_l = torch.tensor(dual_per_layer[l], dtype=torch.float32)
        x_prev = torch.tensor(lp_x_primals[l], dtype=torch.float32)

        with torch.no_grad():
            z_l = layer(x_prev)
        z_l_diff = layer(x_prev.detach())

        z_l_st = z_l_diff - z_l_diff.detach() + z_l.detach()
        proxy = proxy + (nu_l * z_l_st).sum()

    lp_val_tensor = torch.tensor(lp_value, dtype=torch.float32)
    lp_val_differentiable = proxy - proxy.detach() + lp_val_tensor

    if sense == "max":
        gap = lp_val_differentiable - f_theta_val
    else:
        gap = f_theta_val - lp_val_differentiable

    return gap.squeeze()


class LPRelaxationGapRegularizer(BaseRegularizer):
    """Penalizes the LP relaxation gap at training samples.

    For each sample x_i, computes:
        gap(x_i) = LP_max(x_i) - f_theta(x_i)   if sense='max'
        gap(x_i) = f_theta(x_i) - LP_min(x_i)   if sense='min'
        gap(x_i) = LP_max(x_i) - LP_min(x_i)    if sense='both'

    and returns their mean as the regularization loss.

    Parameters
    ----------
    sense : 'max', 'min', or 'both'
    n_samples : int or None
        Number of samples per batch to use (subsampling for speed).
    """

    name = "lp_gap"

    def __init__(self, sense: str = "max", n_samples: Optional[int] = None):
        if sense not in ("max", "min", "both"):
            raise ValueError("sense must be 'max', 'min', or 'both'")
        self.sense = sense
        self.n_samples = n_samples

    def __call__(
        self,
        model: "ReLUNet",
        input_lb: torch.Tensor,
        input_ub: torch.Tensor,
        X_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if X_batch is None:
            return torch.tensor(0.0)

        n = X_batch.shape[0]
        if self.n_samples is not None and self.n_samples < n:
            idx = torch.randperm(n)[: self.n_samples]
            X = X_batch[idx]
        else:
            X = X_batch

        lb_np = input_lb.detach().cpu().numpy()
        ub_np = input_ub.detach().cpu().numpy()
        wb_numpy = model.get_weights_and_biases()
        layer_bounds = interval_bound_propagation_numpy(wb_numpy, lb_np, ub_np)

        gaps = []
        for i in range(X.shape[0]):
            x_i = X[i]

            if self.sense == "both":
                f_theta = model(x_i.unsqueeze(0)).squeeze()
                g_max = _lp_gap_differentiable(
                    model, x_i, layer_bounds, wb_numpy, f_theta, "max",
                )
                g_min = _lp_gap_differentiable(
                    model, x_i, layer_bounds, wb_numpy, f_theta, "min",
                )
                if g_max is not None and g_min is not None:
                    gaps.append(g_max + g_min)
            else:
                f_theta = model(x_i.unsqueeze(0)).squeeze()
                g = _lp_gap_differentiable(
                    model, x_i, layer_bounds, wb_numpy, f_theta, self.sense,
                )
                if g is not None:
                    gaps.append(g)

        if not gaps:
            return torch.tensor(0.0)

        return torch.stack(gaps).mean()


# ========================================================================== #
#  Registry                                                                   #
# ========================================================================== #

def get_regularizer(name: str, **kwargs) -> BaseRegularizer:
    """Get a regularizer by name.

    Args:
        name: one of 'none', 'l1', 'l2', 'bound_width', 'stable_neuron',
              'lp_gap_max', 'lp_gap_min', 'lp_gap_both'
        **kwargs: forwarded to the regularizer constructor (e.g. n_samples for lp_gap)

    Returns:
        Regularizer instance
    """
    if name.startswith("lp_gap"):
        sense = name.split("_")[-1]
        if sense not in ("max", "min", "both"):
            sense = "max"
        return LPRelaxationGapRegularizer(sense=sense, **kwargs)

    registry = {
        "none": BaseRegularizer,
        "l1": L1Regularizer,
        "l2": L2Regularizer,
        "bound_width": BoundWidthRegularizer,
        "stable_neuron": StableNeuronRegularizer,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown regularizer: '{name}'. "
            f"Choose from {list(registry.keys())} or 'lp_gap_max/min/both'."
        )
    return registry[name]()
