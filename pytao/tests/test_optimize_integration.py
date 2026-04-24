"""
Integration tests for :mod:`pytao.optimize`.

These require a working Tao shared library / binary. If pytao cannot load Tao
(``TaoSharedLibraryNotFoundError`` at import of an actual instance), the whole
module is skipped. We also skip if scipy is not installed — the scipy adapter
is the part we most want to exercise here.

We reuse the packaged ``optics_matching_tweaked`` input rather than pointing
at ``$ACC_ROOT_DIR`` so these tests can run anywhere the binary is available.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("scipy")

from pytao import SubprocessTao, Tao  # noqa: E402
from pytao.errors import TaoSharedLibraryNotFoundError  # noqa: E402
from pytao.optimize import (  # noqa: E402
    TaoOptimizationProblem,
    run_scipy_least_squares,
    run_scipy_minimize,
)


PACKAGED_INIT = (
    pathlib.Path(__file__).resolve().parent
    / "input_files"
    / "optics_matching_tweaked"
    / "tao.init"
)


@pytest.fixture(scope="module")
def _tao_available() -> bool:
    """
    Probe once whether a Tao shared library is reachable. If not, every test
    in this module is skipped — we don't want to emit the error repeatedly.
    """
    if not PACKAGED_INIT.exists():
        pytest.skip(f"Packaged init file missing: {PACKAGED_INIT}")
    try:
        tao = Tao(init_file=str(PACKAGED_INIT), noplot=True)
    except TaoSharedLibraryNotFoundError:
        pytest.skip("Tao shared library not available on this host")
    except Exception as exc:  # any other init failure → skip, not fail
        pytest.skip(f"Tao init failed: {exc}")
    # Close if possible, otherwise drop it and let GC handle it.
    close = getattr(tao, "close_subprocess", None)
    if close:
        close()
    return True


@pytest.fixture(params=[Tao, SubprocessTao], ids=["Tao", "SubprocessTao"])
def live_tao(request, _tao_available):
    cls = request.param
    tao = cls(init_file=str(PACKAGED_INIT), noplot=True)
    yield tao
    close = getattr(tao, "close_subprocess", None)
    if close:
        close()


# ---- problem extraction ------------------------------------------------


def test_extracts_quad_variables(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    v1_names = {v.v1_name for v in problem.variables}
    assert v1_names == {"quad"}
    # optics_matching_tweaked searches for Quad::* — expect several active.
    assert problem.n_var > 0


def test_extracts_twiss_datums(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    d2_names = {d.d2_name for d in problem.datums}
    assert d2_names == {"twiss"}
    # end d1 has 6 target datums; max d1 has 2 limit-type datums; all active.
    merit_types = {d.merit_type for d in problem.datums}
    assert {"target", "max", "abs_max"}.issubset(merit_types)
    assert problem.n_data >= 6


def test_evaluate_merit_matches_tao_merit(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    from_problem = problem.evaluate_merit(problem.x0)
    from_tao = live_tao.merit()
    assert from_problem == pytest.approx(from_tao, rel=1e-12, abs=1e-12)


def test_residuals_sum_of_squares_equals_merit(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    r = problem.evaluate_residuals(problem.x0)
    np.testing.assert_allclose(np.sum(r**2), problem.evaluate_merit(problem.x0), rtol=1e-9)


# ---- scipy drives a real optimization ----------------------------------


def test_minimize_reduces_merit(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    m0 = problem.evaluate_merit(problem.x0)
    if m0 == 0.0:
        pytest.skip("Starting merit is already zero for this lattice")
    result = run_scipy_minimize(
        problem,
        method="L-BFGS-B",
        options={"maxiter": 200, "gtol": 1e-10},
    )
    assert result.fun < m0
    # Tao must end up at the solver's final point.
    np.testing.assert_allclose(live_tao.merit(), result.fun, rtol=1e-9)


def test_least_squares_reduces_merit(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    m0 = problem.evaluate_merit(problem.x0)
    if m0 == 0.0:
        pytest.skip("Starting merit is already zero for this lattice")
    result = run_scipy_least_squares(problem, method="trf", max_nfev=200)
    # cost = 0.5 * merit
    assert 2.0 * result.cost < m0


def test_minimize_respects_bounds_when_present(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    # Force tight bounds on the first variable well away from its start.
    v0 = problem.variables[0]
    # mutate in-place: dataclass is frozen, so rebuild the list.
    from pytao.optimize.problem import VariableInfo

    problem.variables[0] = VariableInfo(
        name=v0.name,
        v1_name=v0.v1_name,
        index=v0.index,
        initial_value=v0.initial_value,
        low_lim=v0.initial_value + 0.01,
        high_lim=v0.initial_value + 0.02,
        step=v0.step,
        weight=v0.weight,
        merit_type=v0.merit_type,
        ele_name=v0.ele_name,
        attrib_name=v0.attrib_name,
    )
    result = run_scipy_minimize(problem, method="L-BFGS-B", options={"maxiter": 100})
    assert result.x[0] >= v0.initial_value + 0.01 - 1e-8
    assert result.x[0] <= v0.initial_value + 0.02 + 1e-8


def test_reset_restores_starting_state(live_tao):
    problem = TaoOptimizationProblem(live_tao)
    m0 = problem.evaluate_merit(problem.x0)
    # Perturb.
    problem.set_variables(problem.x0 + 0.01)
    assert problem.evaluate_merit(problem.x0 + 0.01) != pytest.approx(m0, abs=1e-12)
    problem.reset()
    assert live_tao.merit() == pytest.approx(m0, rel=1e-12, abs=1e-12)


# ---- bound translation on a real Tao ----------------------------------


def test_unbounded_limits_translate_to_inf(live_tao):
    """
    The packaged init leaves ``default_low_lim`` / ``default_high_lim``
    commented out, so Tao fills in its ±1e30 sentinels. The problem wrapper
    must translate those to ±inf so scipy's bounded methods handle them.
    """
    problem = TaoOptimizationProblem(live_tao)
    lb, ub = problem.bounds_array
    assert (np.isinf(lb) | (lb > -1e20)).all()
    assert (np.isinf(ub) | (ub < 1e20)).all()
    # At least one of the limits should have been translated to inf for this
    # particular init file (defaults commented out).
    assert np.isinf(lb).any() or np.isinf(ub).any()
