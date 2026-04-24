"""
Unit tests for :mod:`pytao.optimize.scipy_adapter`.

These build a FakeTao whose merit is a genuine quadratic in the variable
vector, so :func:`scipy.optimize.minimize` and :func:`scipy.optimize.least_squares`
actually converge and we can assert on the result.

scipy is an optional dep of pytao; if it's missing the whole module skips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("scipy")

from pytao.optimize import TaoOptimizationProblem  # noqa: E402
from pytao.optimize.scipy_adapter import (  # noqa: E402
    run_scipy_least_squares,
    run_scipy_minimize,
)


# ---- FakeTao with a quadratic forward model ----------------------------


@dataclass
class QuadraticFakeTao:
    """
    Minimal FakeTao where every datum is a linear function of the variables.

    Model: ``y_i = a_i @ x + b_i`` for each datum, with Jacobian rows = ``a_i``.
    Merit = ``sum_i w_i * (y_i - t_i)^2``, so the minimum sits at the
    solution of the weighted least-squares normal equations.
    """

    # Variable setup
    var_values: dict[str, float] = field(default_factory=dict)
    var_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    var_order: list[str] = field(default_factory=list)

    # Datum setup
    datum_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    datum_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    datum_targets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    datum_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    datum_merit_types: list[str] = field(default_factory=list)

    commands: list[str] = field(default_factory=list)

    # ---- helpers --------------------------------------------------------

    def _x_vector(self) -> np.ndarray:
        return np.array([self.var_values[n] for n in self.var_order], dtype=float)

    def _model_values(self) -> np.ndarray:
        x = self._x_vector()
        return self.datum_coeffs @ x + self.datum_offsets

    def _merit_value(self) -> float:
        y = self._model_values()
        total = 0.0
        for yi, ti, wi, mt in zip(
            y, self.datum_targets, self.datum_weights, self.datum_merit_types, strict=True
        ):
            d = _delta_for(mt, yi, ti)
            total += wi * d * d
        return float(total)

    # ---- protocol methods ----------------------------------------------

    def cmd(self, cmd: str, raises: bool = True) -> list[str]:  # noqa: ARG002
        self.commands.append(cmd)
        if cmd.startswith("set var "):
            body = cmd.removeprefix("set var ").strip()
            left, _, rhs = body.partition("=")
            qname, _, _ = left.strip().partition("|")
            # qname comes in as "q[i]"; translate back to the original
            # variable name stored in ``var_order``.
            ix = int(qname.removeprefix("q[").removesuffix("]"))
            self.var_values[self.var_order[ix - 1]] = float(rhs.strip())
        return []

    def var_general(self, *, raises: bool = True):  # noqa: ARG002
        # Group all variables under a single v1 "q".
        return [{"name": "q", "line": "", "lbound": 1, "ubound": len(self.var_order)}]

    def var_v_array(self, v1_var: str, *, raises: bool = True):  # noqa: ARG002
        out = []
        for i, name in enumerate(self.var_order, start=1):
            out.append(
                {
                    "ix_v1": i,
                    "var_attrib_name": "k1",
                    "meas_value": 0.0,
                    "model_value": self.var_values[name],
                    "design_value": self.var_values[name],
                    "useit_opt": True,
                    "good_user": True,
                    "weight": 0.0,
                }
            )
        return out

    def var(self, var: str, *, raises: bool = True):  # noqa: ARG002
        ix = int(var.removeprefix("q[").removesuffix("]"))
        name = self.var_order[ix - 1]
        low, high = self.var_limits.get(name, (-1e30, 1e30))
        return {
            "model_value": self.var_values[name],
            "low_lim": low,
            "high_lim": high,
            "step": 1e-4,
            "weight": 0.0,
            "merit_type": "target",
            "ele_name": name,
            "attrib_name": "k1",
        }

    def data_d2_array(self, ix_uni: str = "", *, raises: bool = True):  # noqa: ARG002
        return ["d"]

    def data_d1_array(self, d2_datum: str, *, raises: bool = True):  # noqa: ARG002
        return [{"name": "end"}]

    def data_d_array(
        self, d2_name: str, d1_name: str, *, ix_uni: str = "", raises: bool = True
    ):  # noqa: ARG002
        y = self._model_values()
        rows = []
        for i, (yi, ti, wi, mt) in enumerate(
            zip(
                y, self.datum_targets, self.datum_weights, self.datum_merit_types, strict=True
            ),
            start=1,
        ):
            rows.append(
                {
                    "ix_d1": i,
                    "data_type": "lin",
                    "merit_type": mt,
                    "ele_ref_name": "",
                    "ele_start_name": "",
                    "ele_name": f"D{i}",
                    "meas_value": float(ti),
                    "model_value": float(yi),
                    "design_value": 0.0,
                    "useit_opt": True,
                    "useit_plot": True,
                    "good_user": True,
                    "weight": float(wi),
                    "exists": True,
                }
            )
        return rows

    def merit(self, *, raises: bool = True) -> float:  # noqa: ARG002
        return self._merit_value()

    def derivative(self, *, raises: bool = True):  # noqa: ARG002
        return {1: self.datum_coeffs.copy()}


def _delta_for(merit_type: str, model: float, target: float) -> float:
    if merit_type == "target":
        return model - target
    if merit_type == "min":
        return min(model - target, 0.0)
    if merit_type == "max":
        return max(model - target, 0.0)
    if merit_type == "abs_min":
        return min(abs(model) - target, 0.0)
    if merit_type == "abs_max":
        return max(abs(model) - target, 0.0)
    return model - target


# ---- builders ----------------------------------------------------------


def _make_two_var_problem(
    limits_1: tuple[float, float] = (-1e30, 1e30),
    limits_2: tuple[float, float] = (-1e30, 1e30),
    initial: tuple[float, float] = (2.5, -1.5),
) -> tuple[QuadraticFakeTao, TaoOptimizationProblem]:
    """
    Linear targets: y1 = x1 (target 1.0), y2 = x2 (target -0.5). Quadratic
    merit has a unique minimum at ``(1.0, -0.5)``.
    """
    tao = QuadraticFakeTao(
        var_values={"x1": initial[0], "x2": initial[1]},
        var_limits={"x1": limits_1, "x2": limits_2},
        var_order=["x1", "x2"],
        datum_coeffs=np.array([[1.0, 0.0], [0.0, 1.0]]),
        datum_offsets=np.zeros(2),
        datum_targets=np.array([1.0, -0.5]),
        datum_weights=np.array([1.0, 1.0]),
        datum_merit_types=["target", "target"],
    )
    return tao, TaoOptimizationProblem(tao)


# ---- minimize ----------------------------------------------------------


def test_minimize_converges_on_quadratic():
    tao, problem = _make_two_var_problem()
    m0 = problem.evaluate_merit(problem.x0)
    result = run_scipy_minimize(problem, method="L-BFGS-B", options={"gtol": 1e-10})
    assert result.fun < m0
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-5)


def test_minimize_respects_bounds():
    # Box the solution out of reach: solution wants x1 = 1.0 but low_lim = 1.5.
    tao, problem = _make_two_var_problem(limits_1=(1.5, 5.0))
    result = run_scipy_minimize(problem, method="L-BFGS-B")
    assert result.x[0] >= 1.5 - 1e-8
    # x2 unbounded should still hit -0.5.
    np.testing.assert_allclose(result.x[1], -0.5, atol=1e-5)


def test_minimize_drops_bounds_for_unbounded_method(caplog):
    tao, problem = _make_two_var_problem(limits_1=(1.5, 5.0))
    with caplog.at_level("INFO", logger="pytao.optimize.scipy_adapter"):
        result = run_scipy_minimize(problem, method="BFGS")
    # method='BFGS' doesn't honour bounds → x1 reaches 1.0 freely.
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-5)
    assert any("does not accept bounds" in rec.message for rec in caplog.records)


def test_minimize_leaves_tao_at_solution():
    tao, problem = _make_two_var_problem()
    result = run_scipy_minimize(problem, method="L-BFGS-B")
    # The last ``set var`` issued by the adapter must encode result.x.
    last = [c for c in tao.commands if c.startswith("set var q[")]
    assert "|model = " in last[-1]
    # Tao's internal state should now give the final merit.
    np.testing.assert_allclose(tao.merit(), result.fun, rtol=1e-9)


def test_minimize_honours_soft_max_constraint():
    """A 'max'-type datum lets scipy push through until the bound bites."""
    tao = QuadraticFakeTao(
        var_values={"x1": 0.0},
        var_limits={"x1": (-1e30, 1e30)},
        var_order=["x1"],
        datum_coeffs=np.array([[1.0], [1.0]]),
        datum_offsets=np.zeros(2),
        datum_targets=np.array([10.0, 3.0]),  # want x1 = 10 but capped at 3
        datum_weights=np.array([1.0, 1e6]),  # huge weight on the cap
        datum_merit_types=["target", "max"],
    )
    problem = TaoOptimizationProblem(tao)
    result = run_scipy_minimize(problem, method="L-BFGS-B", options={"gtol": 1e-10})
    # Soft cap with huge weight should hold x1 near 3.0.
    assert result.x[0] == pytest.approx(3.0, abs=1e-3)


def test_minimize_accepts_explicit_x0():
    tao, problem = _make_two_var_problem(initial=(2.5, -1.5))
    x0 = np.array([0.2, 0.2])
    result = run_scipy_minimize(problem, method="L-BFGS-B", x0=x0)
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-5)


# ---- least_squares -----------------------------------------------------


def test_least_squares_converges():
    tao, problem = _make_two_var_problem()
    result = run_scipy_least_squares(problem, method="trf")
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-6)
    # ``cost`` = 0.5 * sum(r^2) = 0.5 * merit ≈ 0
    assert result.cost < 1e-12


def test_least_squares_with_analytic_jacobian():
    tao, problem = _make_two_var_problem()
    result = run_scipy_least_squares(problem, method="trf", use_jacobian=True)
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-6)


def test_least_squares_respects_bounds():
    # Start inside the box; optimum wants x2 = -0.5 but the lower bound is 0.
    tao, problem = _make_two_var_problem(limits_2=(0.0, 5.0), initial=(2.5, 1.5))
    result = run_scipy_least_squares(problem, method="trf")
    # x2 is pinned to the lower bound.
    assert result.x[1] == pytest.approx(0.0, abs=1e-6)


def test_least_squares_lm_logs_bounds_ignore(caplog):
    tao, problem = _make_two_var_problem(limits_1=(0.0, 5.0))
    with caplog.at_level("INFO", logger="pytao.optimize.scipy_adapter"):
        run_scipy_least_squares(problem, method="lm")
    assert any("ignores bounds" in rec.message for rec in caplog.records)


def test_least_squares_raises_without_datums():
    tao, problem = _make_two_var_problem()
    problem.datums = []  # type: ignore[misc]
    # Use object.__setattr__ in case datums is frozen in the future.
    object.__setattr__(problem, "datums", [])
    with pytest.raises(ValueError, match="at least one active datum"):
        run_scipy_least_squares(problem)


# ---- adapter sanity -----------------------------------------------------


def test_minimize_callable_matches_evaluate_merit():
    tao, problem = _make_two_var_problem()
    f = problem.scipy_minimize_callable()
    x = np.array([0.25, 0.75])
    assert f(x) == problem.evaluate_merit(x)


def test_residuals_callable_matches_evaluate_residuals():
    tao, problem = _make_two_var_problem()
    r = problem.scipy_residuals_callable()
    x = np.array([0.25, 0.75])
    np.testing.assert_array_equal(r(x), problem.evaluate_residuals(x))


# ---- docs snippet regression: differential_evolution ---------------------


def test_differential_evolution_snippet_works():
    """
    docs/optimize.md shows calling scipy.optimize.differential_evolution
    directly (not via minimize(method=...)). Verify the documented pattern.
    """
    from scipy.optimize import differential_evolution

    tao, problem = _make_two_var_problem(
        limits_1=(-5.0, 5.0), limits_2=(-5.0, 5.0), initial=(0.0, 0.0)
    )
    result = differential_evolution(
        problem.evaluate_merit,
        bounds=problem.bounds,
        x0=problem.x0,
        tol=1e-8,
        seed=0,
        polish=True,
        maxiter=200,
    )
    problem.set_variables(result.x)
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-3)
