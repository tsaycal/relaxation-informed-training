"""
Benchmark optimization test functions.

Each function is defined over a standard domain and can generate
training/validation data via Latin Hypercube or uniform random sampling.

Benchmarks follow Plate et al. (2026).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BenchmarkFunction:
    """Container for a benchmark function with its domain."""
    name: str
    dim: int
    domain: np.ndarray  # shape (dim, 2), each row is [lb, ub]
    func: callable
    known_minimum: Optional[float] = None
    known_minimizer: Optional[np.ndarray] = None


def peaks(x: np.ndarray) -> np.ndarray:
    """MATLAB Peaks function (Plate et al. 2026, Eq. 19).

    A two-variable function with three local maxima and three local minima.
    Domain: [-2, 2]^2.
    """
    x1, x2 = x[:, 0], x[:, 1]
    return (
        3.0 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2)
        - 10.0 * (x1 / 5.0 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2)
        - (1.0 / 3.0) * np.exp(-(x1 + 1)**2 - x2**2)
    )


def himmelblau(x: np.ndarray) -> np.ndarray:
    """Himmelblau's function (Plate et al. 2026, Eq. 21).

    Has four identical local minima, each with f = 0.
    Domain: [-5, 5]^2.
    """
    x1, x2 = x[:, 0], x[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley function. Global min f=0 at x=(0,...,0).

    Domain: [-3.5, 3.5]^d (following Plate et al. 2026).
    """
    dim = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
        - np.exp(sum_cos / dim)
        + 20.0 + np.e
    )


def get_benchmark(name: str, dim: int = 2) -> BenchmarkFunction:
    """Get a benchmark function by name.

    Args:
        name: one of 'peaks', 'himmelblau', 'ackley'
        dim: input dimension (peaks and himmelblau require dim=2)
    """
    benchmarks = {
        "peaks": {
            "func": peaks,
            "domain": np.array([[-2.0, 2.0], [-2.0, 2.0]]),
            "known_minimum": -6.55,
            "known_minimizer": None,
        },
        "himmelblau": {
            "func": himmelblau,
            "domain": np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            "known_minimum": 0.0,
            "known_minimizer": np.array([3.0, 2.0]),
        },
        "ackley": {
            "func": ackley,
            "domain": np.array([[-3.5, 3.5]] * dim),
            "known_minimum": 0.0,
            "known_minimizer": np.zeros(dim),
        },
    }

    if name not in benchmarks:
        raise ValueError(f"Unknown benchmark: {name}. Choose from {list(benchmarks.keys())}")

    if name in ("peaks", "himmelblau") and dim != 2:
        raise ValueError(f"Benchmark '{name}' is only defined for dim=2, got dim={dim}")

    info = benchmarks[name]
    return BenchmarkFunction(
        name=name,
        dim=dim,
        domain=info["domain"],
        func=info["func"],
        known_minimum=info["known_minimum"],
        known_minimizer=info["known_minimizer"],
    )


def generate_data(
    benchmark: BenchmarkFunction,
    n_samples: int,
    method: str = "lhs",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data from a benchmark function.

    Args:
        benchmark: BenchmarkFunction instance
        n_samples: number of samples
        method: 'lhs' for Latin Hypercube Sampling, 'uniform' for uniform random
        seed: random seed
    """
    rng = np.random.RandomState(seed)
    dim = benchmark.dim
    lb = benchmark.domain[:, 0]
    ub = benchmark.domain[:, 1]

    if method == "lhs":
        samples = np.zeros((n_samples, dim))
        for j in range(dim):
            perm = rng.permutation(n_samples)
            samples[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples
        X = lb + samples * (ub - lb)
    elif method == "uniform":
        X = lb + rng.uniform(size=(n_samples, dim)) * (ub - lb)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    y = benchmark.func(X)
    return X, y
