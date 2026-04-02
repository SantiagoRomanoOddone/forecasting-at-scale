"""Hyperparameter tuning with Optuna."""

import logging
from typing import Any

import optuna
from optuna.samplers import TPESampler

from forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


def sample_params(trial: optuna.Trial, space: dict) -> dict:
    """Sample hyperparameters from the defined search space.

    Args:
        trial: Optuna trial object.
        space: Dict mapping param names to (type, *args).

    Returns:
        Dict of sampled hyperparameters.
    """
    params = {}
    for name, definition in space.items():
        param_type = definition[0]
        args = definition[1:]

        if param_type == "int":
            params[name] = trial.suggest_int(name, *args)
        elif param_type == "float":
            params[name] = trial.suggest_float(name, *args)
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, *args)
        else:
            raise ValueError(f"Unknown param type: {param_type}")

    return params


def tune(
    model_class: type[BaseModel],
    train_data: Any,
    val_data: Any,
    prediction_length: int,
    context_length: int,
    n_trials: int = 10,
    **model_kwargs,
) -> dict:
    """Run Bayesian hyperparameter search.

    Args:
        model_class: The model class to tune.
        train_data: Training data.
        val_data: Validation data.
        prediction_length: Forecast horizon.
        context_length: Input window size.
        n_trials: Number of Optuna trials.
        **model_kwargs: Extra kwargs passed to the model constructor.

    Returns:
        Dict with "best_params" and "best_value" keys.
    """
    # Get search space from a dummy instance
    dummy = model_class(
        prediction_length=prediction_length,
        context_length=context_length,
        **model_kwargs,
    )
    space = dummy.get_hyperparameter_space()

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)

        model = model_class(
            prediction_length=prediction_length,
            context_length=context_length,
            **model_kwargs,
            **params,
        )
        model.fit(train_data, val_data)
        results = model.predict(val_data)

        # RMSE as objective
        rmse = ((results["prediction"] - results["actual"]) ** 2).mean() ** 0.5
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best trial: RMSE=%.4f | params=%s",
        study.best_value,
        study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
