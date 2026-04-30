"""Training visualizer package."""

__version__ = "1.0.0"
__all__ = ["visualize_training"]


def __getattr__(name):
    if name == "visualize_training":
        from .core.visualizer import visualize_training
        return visualize_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
