"""Suggested param_grid for DANetModule hyperparameter tuning."""

import optuna


def get_danet_param_grid(
    input_dim: int,
    small_search: bool = False,
) -> dict:
    """Generate a param_grid for DANetModule compatible with HyperparameterTuner.

    Parameters
    ----------
    input_dim : int
        Number of input features (needed to validate hidden_dims divisibility).
    small_search : bool
        If True, return a minimal grid for quick prototyping (~10 trials).
        If False, return a full grid for thorough search (~50-100 trials).

    Returns
    -------
    dict
        param_grid ready for HyperparameterTuner(..., param_grid=param_grid).
    """

    # Ensure hidden_dims[0] is divisible by attention_heads
    # We handle this in the pipeline by rounding or asserting,
    # but here we restrict to safe values.
    safe_heads = [2, 4, 8] if input_dim >= 16 else [2, 4]

    if small_search:
        return {
            # Architecture
            "hidden_dims": [
                [128, 64],
                [256, 128, 64],
            ],
            "dropout": [0.1, 0.3, 0.5],
            "attention_heads": safe_heads,
            "use_sample_attention": [False, True],
        }

    # ------------------------------------------------------------------
    # Full search space — mix of discrete lists and Optuna distributions
    # ------------------------------------------------------------------
    return {
        # --- Architecture ---
        # List of hidden layer widths. First element must be divisible by
        # attention_heads (enforced in DANetModule via assert).
        "hidden_dims": [
            [128, 64],
            [128, 64, 32],
            [256, 128],
            [256, 128, 64],
            [256, 128, 64, 32],
            [512, 256, 128],
        ],

        # --- Regularization ---
        # Dropout rate — continuous search via uniform distribution
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.6),

        # --- Attention ---
        # Number of attention heads — must divide hidden_dims[0]
        "attention_heads": safe_heads,

        # Whether to use the optional sample-wise attention layer
        "use_sample_attention": [False, True],

        # --- Training hyperparameters (if your pipeline exposes them) ---
        # Often pipelines also expose optimizer params — include if available:
        # "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        # "batch_size": [32, 64, 128, 256],
        # "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
    }
