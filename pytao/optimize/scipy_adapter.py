"""
Adapters that run :mod:`scipy.optimize` solvers against a
:class:`TaoOptimizationProblem`.

scipy is an optional dependency; importing this module raises a clear error if
it isn't installed. Install via ``pip install pytao[optimize]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .problem import TaoOptimizationProblem

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

logger = logging.getLogger(__name__)

# scipy's ``minimize`` drivers that accept a ``bounds=`` argument.
_METHODS_SUPPORTING_BOUNDS = frozenset(
    {
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
        "Nelder-Mead",
        "COBYLA",
        "COBYQA",
    }
)


def _require_scipy():
    try:
        import scipy.optimize  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial
        raise ImportError(
            "scipy is required for pytao.optimize's scipy adapters. "
            "Install with `pip install scipy` or `pip install pytao[optimize]`."
        ) from exc


def run_scipy_minimize(
    problem: TaoOptimizationProblem,
    method: str = "L-BFGS-B",
    *,
    use_bounds: bool = True,
    x0: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
    **minimize_kwargs: Any,
) -> OptimizeResult:
    """
    Minimize Tao's merit function with :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    problem : TaoOptimizationProblem
        The problem snapshot. On return, the Tao instance is left at the point
        ``result.x`` found by scipy (not necessarily the global minimum).
    method : str, default="L-BFGS-B"
        scipy method name. Bounded methods get ``problem.bounds`` automatically.
    use_bounds : bool, default=True
        If ``False``, run the solver unbounded even when ``method`` would accept
        bounds. Useful for benchmarking or when your init file uses the default
        ±1e30 limits.
    x0 : ndarray, optional
        Starting point. Defaults to ``problem.x0`` (the variable values at the
        time the problem was built).
    options : dict, optional
        Passed through as ``minimize(..., options=options)``.
    **minimize_kwargs
        Any remaining keyword arguments are forwarded to
        :func:`scipy.optimize.minimize` (``jac``, ``tol``, ``callback`` …).

    Returns
    -------
    scipy.optimize.OptimizeResult
        scipy's result object. ``result.x`` holds the final variable vector;
        ``result.fun`` holds the final Tao merit.

    Notes
    -----
    scipy methods that support analytic gradients (``CG``, ``BFGS``, ``Newton-CG``,
    ``L-BFGS-B``, ``TNC``, ``SLSQP``, ``trust-ncg``, ``trust-krylov``,
    ``trust-exact``, ``trust-constr``) will benefit from passing
    ``jac=_merit_gradient_from_jacobian`` — see below for a ready-made callback.
    We do *not* pass a gradient by default because it requires Tao to have
    computed a fresh ``derivative()`` matrix, which isn't always valid for
    every init setup.
    """
    _require_scipy()
    from scipy.optimize import minimize

    if x0 is None:
        x0 = problem.x0
    bounds = problem.bounds if (use_bounds and method in _METHODS_SUPPORTING_BOUNDS) else None
    if use_bounds and method not in _METHODS_SUPPORTING_BOUNDS:
        logger.info("Method %r does not accept bounds; dropping them.", method)

    logger.info(
        "Running scipy.optimize.minimize(method=%r) with %d variables, %d datums",
        method,
        problem.n_var,
        problem.n_data,
    )
    result = minimize(
        fun=problem.evaluate_merit,
        x0=x0,
        method=method,
        bounds=bounds,
        options=options,
        **minimize_kwargs,
    )
    # Ensure Tao ends up at the solver's final point — minimize may have
    # evaluated extra trial steps after the accepted one.
    problem.set_variables(result.x)
    return result


def run_scipy_least_squares(
    problem: TaoOptimizationProblem,
    method: str = "trf",
    *,
    use_bounds: bool = True,
    use_jacobian: bool = False,
    x0: np.ndarray | None = None,
    **ls_kwargs: Any,
) -> OptimizeResult:
    """
    Minimize Tao's merit with :func:`scipy.optimize.least_squares`.

    This treats each datum's weighted delta as a residual, giving nonlinear
    least-squares solvers the structure they want. For Levenberg-Marquardt-like
    problems (most Tao optimizations) this is typically faster and more robust
    than generic ``minimize``.

    Parameters
    ----------
    problem : TaoOptimizationProblem
        The problem snapshot.
    method : str, default="trf"
        One of ``"trf"``, ``"dogbox"``, or ``"lm"`` (``"lm"`` does not support
        bounds).
    use_bounds : bool, default=True
        If ``True`` and ``method != "lm"``, pass ``problem.bounds_array``.
    use_jacobian : bool, default=False
        If ``True``, pass :meth:`TaoOptimizationProblem.residual_jacobian` as
        the analytic Jacobian. This requires Tao's ``derivative()`` to be
        computable — for many datums this is fine, but some exotic merit types
        may not be differentiable by Tao. Default is to let scipy use finite
        differences.
    x0 : ndarray, optional
        Starting point. Defaults to ``problem.x0``.
    **ls_kwargs
        Forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``result.x`` is the solution, ``result.fun`` is the residual vector,
        ``result.cost = 0.5 * sum(result.fun**2)`` equals half Tao's merit.
    """
    _require_scipy()
    from scipy.optimize import least_squares

    if problem.n_data == 0:
        raise ValueError(
            "run_scipy_least_squares needs at least one active datum; the problem has none."
        )
    if x0 is None:
        x0 = problem.x0

    if use_bounds and method != "lm":
        bounds = problem.bounds_array
    else:
        if use_bounds and method == "lm":
            logger.info("method='lm' ignores bounds; passing ±inf.")
        bounds = (-np.inf, np.inf)

    jac: Any = "2-point"
    if use_jacobian:
        jac = lambda x: problem.residual_jacobian(x)  # noqa: E731

    logger.info(
        "Running scipy.optimize.least_squares(method=%r) with %d variables, %d datums",
        method,
        problem.n_var,
        problem.n_data,
    )
    result = least_squares(
        fun=problem.evaluate_residuals,
        x0=x0,
        jac=jac,
        bounds=bounds,
        method=method,
        **ls_kwargs,
    )
    problem.set_variables(result.x)
    return result
