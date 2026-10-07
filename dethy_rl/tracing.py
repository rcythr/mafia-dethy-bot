"""Thin helpers so MLflow tracing is a no-op unless enabled (and never imported otherwise)."""
from contextlib import nullcontext


def span(name: str, span_type: str, enabled: bool):
    """Context manager yielding an MLflow span, or None when tracing is off."""
    if not enabled:
        return nullcontext(None)
    import mlflow

    return mlflow.start_span(name=name, span_type=span_type)
