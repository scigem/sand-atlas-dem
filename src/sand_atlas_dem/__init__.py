"""GPU LS-DEM research implementation; numerical evidence is separate from validation."""

__version__ = "0.1.0a1"

from .geometry import ShapeLibrary
from .spec import Experiment

__all__ = ["ShapeLibrary", "Experiment", "Simulation", "RunResult"]


def __getattr__(name):
    if name in {"Simulation", "RunResult"}:
        from .simulation import RunResult, Simulation

        return {"Simulation": Simulation, "RunResult": RunResult}[name]
    raise AttributeError(name)
