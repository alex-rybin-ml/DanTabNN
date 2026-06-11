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
    safe_heads = [2, 4, 8] if input_dim >= 16 else [2, 4]

    if small_search:
        return {
            "dropout": [0.1, 0.3],
            "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
            "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
            "hidden_dims_choice": ["adaptive"],
            "gating_type": ["soft", "none"],
        }

    return {
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
        "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        "hidden_dims_choice": ["adaptive", "narrow", "wide"],
        "gating_type": ["soft", "none"],
    }


def _compute_hidden_dims(n_features: int):
    """Adaptive hidden layer dimensions (same as experiment runner)."""
    h0 = max(32, min(n_features * 2, 128))
    h0 = ((h0 + 3) // 4) * 4
    h1 = max(16, min(n_features, 64))
    h1 = ((h1 + 3) // 4) * 4
    h2 = max(8, min(n_features // 2, 32))
    h2 = ((h2 + 3) // 4) * 4
    return [h0, h1, h2]


def get_danet_param_mapper(params: dict, pipeline) -> dict:
    """Map symbolic param names to pipeline constructor kwargs.

    Translates "hidden_dims_choice" → actual "hidden_dims" list,
    adds default values for gating_k, and preserves all other params.
    """
    n_feat = len(pipeline.numeric_features)
    pipe_kwargs = dict(params)  # copy

    # Map hidden_dims_choice → hidden_dims
    choice = pipe_kwargs.pop("hidden_dims_choice", "adaptive")
    if choice == "adaptive":
        pipe_kwargs["hidden_dims"] = _compute_hidden_dims(n_feat)
    elif choice == "narrow":
        pipe_kwargs["hidden_dims"] = [32, 16, 8]
    elif choice == "wide":
        pipe_kwargs["hidden_dims"] = [256, 128, 64]

    # Add required pipeline kwargs that aren't in the grid
    pipe_kwargs.setdefault("numeric_features", pipeline.numeric_features)
    pipe_kwargs.setdefault("categorical_features", pipeline.categorical_features)
    pipe_kwargs.setdefault("target_column", pipeline.target_column)
    pipe_kwargs.setdefault("gating_k", max(1, n_feat // 3))
    pipe_kwargs.setdefault("batch_size", pipeline.batch_size)
    pipe_kwargs.setdefault("epochs", 100)
    pipe_kwargs.setdefault("early_stopping_patience", 15)
    pipe_kwargs.setdefault("random_state", pipeline.random_state)
    pipe_kwargs.setdefault("n_classes", getattr(pipeline, "n_classes", None))

    return pipe_kwargs
