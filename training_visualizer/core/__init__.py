"""Core visualization functionality."""

__all__ = ["visualize_training", "visualize_training_comparison"]


def __getattr__(name):
    if name == "visualize_training":
        from .visualizer import visualize_training
        return visualize_training
    if name == "visualize_training_comparison":
        from .visualizer import visualize_training_comparison
        return visualize_training_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
