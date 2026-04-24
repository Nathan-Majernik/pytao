"""
Drive Tao optimization from Python-native optimizer libraries.

This subpackage lets you use a running :class:`~pytao.Tao` instance (initialized
the usual way — including from a ``tao.init`` file) as the forward model for
external optimizers such as :mod:`scipy.optimize`.

The workflow is:

1. Build a :class:`TaoOptimizationProblem` from a live Tao instance. The problem
   extracts the set of variables and datums that Tao itself would use for
   optimization (``useit_opt`` flag), along with their bounds, weights, and
   merit types, so weights/objectives/constraints declared in the init file are
   honoured automatically.
2. Hand the problem to an adapter such as :func:`run_scipy_minimize` or
   :func:`run_scipy_least_squares`, or to any external optimizer via the
   problem's callables (``evaluate_merit``, ``evaluate_residuals``,
   ``jacobian``).

At the end of a run the Tao instance is left at the best point found, so
subsequent Tao commands (``show lattice``, ``write``, …) see the optimized
state.
"""

from __future__ import annotations

from .problem import (
    DatumInfo,
    TaoOptimizationProblem,
    VariableInfo,
)
from .scipy_adapter import (
    run_scipy_least_squares,
    run_scipy_minimize,
)

__all__ = [
    "DatumInfo",
    "TaoOptimizationProblem",
    "VariableInfo",
    "run_scipy_least_squares",
    "run_scipy_minimize",
]
