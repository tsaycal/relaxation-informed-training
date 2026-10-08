"""
Training pipeline for ReLU networks with pluggable regularization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from models import ReLUNet
from regularizers import BaseRegularizer


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0  # PyTorch built-in weight decay (separate from our regularizers)
    reg_weight: float = 1e-3   # weight for the custom regularizer
    patience: int = 50         # early stopping patience
    min_delta: float = 1e-6    # minimum improvement for early stopping
    seed: int = 42


@dataclass
class TrainResult:
    """Training result container."""
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    reg_losses: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    final_train_loss: float = float("inf")
    final_val_loss: float = float("inf")
    # Target normalization parameters (set when normalize_y=True)
    y_min: float = 0.0
    y_range: float = 1.0


def train_model(
    model: ReLUNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    regularizer: BaseRegularizer,
    config: TrainConfig,
    normalize_y: bool = True,
    verbose: bool = True,
    earlystop: bool = False
) -> TrainResult:
    """Train a ReLU network with optional regularization.

    Args:
        model: ReLUNet to train
        X_train, y_train: training data (numpy arrays)
        X_val, y_val: validation data (numpy arrays)
        input_lb, input_ub: input domain bounds (for regularizer)
        regularizer: regularization callable
        config: training configuration
        normalize_y: whether to normalize targets to [0, 1]
        verbose: print progress

    Returns:
        TrainResult with training history
    """
    torch.manual_seed(config.seed)
    device = next(model.parameters()).device

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    lb = torch.tensor(input_lb, dtype=torch.float32, device=device)
    ub = torch.tensor(input_ub, dtype=torch.float32, device=device)

    # Optional target normalization
    if normalize_y:
        y_min, y_max = y_train.min(), y_train.max()
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_tr = torch.tensor(
            (y_train - y_min) / y_range, dtype=torch.float32, device=device
        )
        y_v = torch.tensor(
            (y_val - y_min) / y_range, dtype=torch.float32, device=device
        )
    else:
        y_min, y_range = 0.0, 1.0
        y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Reshape targets
    if y_tr.dim() == 1:
        y_tr = y_tr.unsqueeze(1)
    if y_v.dim() == 1:
        y_v = y_v.unsqueeze(1)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Training loop
    result = TrainResult(y_min=float(y_min), y_range=float(y_range))
    best_state = None
    patience_counter = 0
    mse_loss = nn.MSELoss()

    n_samples = X_tr.shape[0]

    for epoch in range(config.epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        epoch_reg = 0.0
        n_batches = 0

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i : i + config.batch_size]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]

            optimizer.zero_grad()

            # Forward pass
            pred = model(X_batch)
            data_loss = mse_loss(pred, y_batch)

            # Regularization (pass X_batch for LP-based regularizers)
            reg_loss = regularizer(model, lb, ub, X_batch=X_batch)
            total_loss = data_loss + config.reg_weight * reg_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += data_loss.item()
            epoch_reg += reg_loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        avg_reg_loss = epoch_reg / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = mse_loss(val_pred, y_v).item()

        result.train_losses.append(avg_train_loss)
        result.val_losses.append(val_loss)
        result.reg_losses.append(avg_reg_loss)

        # Early stopping
        if val_loss < result.best_val_loss - config.min_delta:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience and earlystop:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and (epoch % 100 == 0 or epoch == config.epochs - 1):
            print(
                f"  Epoch {epoch:4d} | "
                f"Train MSE: {avg_train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Reg: {avg_reg_loss:.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    result.final_train_loss = result.train_losses[-1]
    result.final_val_loss = result.val_losses[-1]

    return result
