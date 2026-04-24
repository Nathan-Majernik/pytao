"""
Python-driven optimization of the ``optics_matching_tweaked`` example.

Runs four optimizations of the same init file — Tao's built-in ``lmdif`` as a
reference, scipy's ``L-BFGS-B``, scipy's ``least_squares`` with finite
differences, and scipy's ``least_squares`` with Tao's analytic Jacobian — and
prints a side-by-side comparison of final merit and number of forward
evaluations.

Run with: ``python docs/examples/optimize.py`` from the pytao repo root.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
from pytao import Tao
from pytao.optimize import (
    TaoOptimizationProblem,
    run_scipy_least_squares,
    run_scipy_minimize,
)

INIT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "pytao"
    / "tests"
    / "input_files"
    / "optics_matching_tweaked"
    / "tao.init"
)


def fresh_tao() -> Tao:
    return Tao(init_file=str(INIT), noplot=True)


def _report(label: str, m0: float, m1: float, nfev: int | None, seconds: float) -> None:
    reduction = 100.0 * (m0 - m1) / max(m0, 1e-30)
    nfev_str = f"{nfev:>4}" if nfev is not None else "  —"
    print(
        f"{label:<35} start={m0:>10.3e}  final={m1:>10.3e}  "
        f"↓{reduction:6.2f}%  nfev={nfev_str}  {seconds * 1000:7.1f} ms"
    )


def run_tao_lmdif() -> None:
    """Baseline: use Tao's own optimizer, driven from Python."""
    tao = fresh_tao()
    m0 = tao.merit()
    t0 = time.perf_counter()
    tao.cmd("set global optimizer = lmdif")
    tao.cmd("set global n_opti_cycles = 100")
    # ``run`` blocks until Tao's optimizer is done.
    tao.cmd("run")
    dt = time.perf_counter() - t0
    _report("Tao built-in lmdif", m0, tao.merit(), None, dt)


def run_scipy_minimize_lbfgs() -> None:
    tao = fresh_tao()
    problem = TaoOptimizationProblem(tao)
    m0 = problem.evaluate_merit(problem.x0)
    t0 = time.perf_counter()
    result = run_scipy_minimize(problem, method="L-BFGS-B", options={"gtol": 1e-10})
    dt = time.perf_counter() - t0
    _report("scipy.optimize.minimize(L-BFGS-B)", m0, result.fun, result.nfev, dt)


def run_scipy_least_squares_fd() -> None:
    tao = fresh_tao()
    problem = TaoOptimizationProblem(tao)
    m0 = problem.evaluate_merit(problem.x0)
    t0 = time.perf_counter()
    result = run_scipy_least_squares(problem, method="trf")
    dt = time.perf_counter() - t0
    _report("scipy.least_squares (fd jac)", m0, 2.0 * float(result.cost), result.nfev, dt)


def run_scipy_least_squares_analytic() -> None:
    tao = fresh_tao()
    problem = TaoOptimizationProblem(tao)
    m0 = problem.evaluate_merit(problem.x0)
    t0 = time.perf_counter()
    result = run_scipy_least_squares(problem, method="trf", use_jacobian=True)
    dt = time.perf_counter() - t0
    _report("scipy.least_squares (tao jac)", m0, 2.0 * float(result.cost), result.nfev, dt)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    print(f"Init file: {INIT}\n")
    run_tao_lmdif()
    run_scipy_minimize_lbfgs()
    run_scipy_least_squares_fd()
    run_scipy_least_squares_analytic()


if __name__ == "__main__":
    main()
