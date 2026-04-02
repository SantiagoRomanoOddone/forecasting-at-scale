"""Model registry: maps string names to model classes."""

from forecasting.models.base import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register(name: str):
    """Decorator to register a model class.

    Usage:
        @register("deepar")
        class DeepARModel(BaseModel):
            ...
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> type[BaseModel]:
    """Retrieve a model class by name.

    Args:
        name: Registered model name (e.g. "deepar", "xgboost").

    Returns:
        The model class.

    Raises:
        KeyError: If model name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found. Available: {available}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_REGISTRY.keys())
