"""
Unit tests for :mod:`pytao.optimize.problem`.

These tests do not require a built Tao binary — they drive the code with a
FakeTao that returns scripted responses matching the structured output of
``var_general``, ``var_v_array``, ``var``, ``data_d*``, ``merit``, and
``derivative``. Integration against a real Tao lives in
``test_optimize_integration.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from pytao.optimize import DatumInfo, TaoOptimizationProblem, VariableInfo
from pytao.optimize.problem import _compute_datum_delta, _finite_limit


# ---- FakeTao ------------------------------------------------------------


@dataclass
class FakeTao:
    """
    A stub that satisfies the ``_TaoLike`` protocol.

    Tests build a FakeTao with scripted responses for ``var_general``,
    ``var_v_array``, per-variable ``var``, and the ``data_d*`` family, plus a
    callable ``merit_fn(x)`` the fake uses to compute merit after the tests
    apply ``set var`` commands.
    """

    var_general_rows: list[dict[str, Any]] = field(default_factory=list)
    var_v_array_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    var_detail: dict[str, dict[str, Any]] = field(default_factory=dict)
    d2_names: list[str] = field(default_factory=list)
    d1_arrays: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    d_arrays: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    merit_fn: Any = None
    derivative_matrix: dict[int, np.ndarray] = field(default_factory=dict)

    # Command log — every cmd() that comes through lands here for assertions.
    commands: list[str] = field(default_factory=list)

    def cmd(self, cmd: str, raises: bool = True) -> list[str]:  # noqa: ARG002
        self.commands.append(cmd)
        # Update the current model value stored in var_detail so subsequent
        # var() lookups reflect the write.
        if cmd.startswith("set var "):
            body = cmd.removeprefix("set var ").strip()
            left, _, rhs = body.partition("=")
            name, _, which = left.strip().partition("|")
            which = which.strip()
            value = float(rhs.strip())
            if name in self.var_detail and which == "model":
                self.var_detail[name] = {**self.var_detail[name], "model_value": value}
        return []

    def var_general(self, *, raises: bool = True) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.var_general_rows)

    def var_v_array(self, v1_var: str, *, raises: bool = True):  # noqa: ARG002
        return list(self.var_v_array_rows.get(v1_var, []))

    def var(self, var: str, *, raises: bool = True):  # noqa: ARG002
        return dict(self.var_detail[var])

    def data_d2_array(self, ix_uni: str = "", *, raises: bool = True):  # noqa: ARG002
        return list(self.d2_names)

    def data_d1_array(self, d2_datum: str, *, raises: bool = True):  # noqa: ARG002
        # d2_datum comes in as "{ix_uni}@{d2_name}"
        _, _, d2_name = d2_datum.partition("@")
        return list(self.d1_arrays.get(d2_name, []))

    def data_d_array(
        self,
        d2_name: str,
        d1_name: str,
        *,
        ix_uni: str = "",  # noqa: ARG002
        raises: bool = True,  # noqa: ARG002
    ):
        return list(self.d_arrays.get((d2_name, d1_name), []))

    def merit(self, *, raises: bool = True) -> float:  # noqa: ARG002
        assert self.merit_fn is not None, "FakeTao.merit_fn not set for this test"
        return float(self.merit_fn(self))

    def derivative(self, *, raises: bool = True):  # noqa: ARG002
        return {k: v.copy() for k, v in self.derivative_matrix.items()}


def _make_var_detail(
    model: float,
    low: float,
    high: float,
    step: float = 1e-4,
    weight: float = 0.0,
    merit_type: str = "target",
    ele_name: str = "Q1",
    attrib_name: str = "k1",
) -> dict[str, Any]:
    return {
        "model_value": model,
        "low_lim": low,
        "high_lim": high,
        "step": step,
        "weight": weight,
        "merit_type": merit_type,
        "ele_name": ele_name,
        "attrib_name": attrib_name,
    }


def _simple_problem_tao() -> FakeTao:
    """
    A realistic 3-variable, 2-datum FakeTao that several tests share.

    - v1 ``quad`` has 3 entries, 2 active (useit_opt True) and 1 inactive.
    - d2 ``twiss`` has a d1 ``end`` with 2 target datums, both active.
    """
    tao = FakeTao()
    tao.var_general_rows = [{"name": "quad", "line": "", "lbound": 1, "ubound": 3}]
    tao.var_v_array_rows = {
        "quad": [
            {
                "ix_v1": 1,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": 0.5,
                "design_value": 0.5,
                "useit_opt": True,
                "good_user": True,
                "weight": 0.0,
            },
            {
                "ix_v1": 2,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": -0.3,
                "design_value": -0.3,
                "useit_opt": True,
                "good_user": True,
                "weight": 0.0,
            },
            {
                "ix_v1": 3,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": 0.1,
                "design_value": 0.1,
                "useit_opt": False,  # inactive — should be filtered out
                "good_user": False,
                "weight": 0.0,
            },
        ]
    }
    tao.var_detail = {
        "quad[1]": _make_var_detail(0.5, -5.0, 5.0, ele_name="Q1"),
        "quad[2]": _make_var_detail(-0.3, -1e30, 1e30, ele_name="Q2"),  # unbounded sentinels
        "quad[3]": _make_var_detail(0.1, -5.0, 5.0, ele_name="Q3"),
    }
    tao.d2_names = ["twiss"]
    tao.d1_arrays = {"twiss": [{"name": "end"}]}
    tao.d_arrays = {
        ("twiss", "end"): [
            {
                "ix_d1": 1,
                "data_type": "beta.a",
                "merit_type": "target",
                "ele_ref_name": "",
                "ele_start_name": "",
                "ele_name": "END",
                "meas_value": 12.5,
                "model_value": 10.0,
                "design_value": 10.0,
                "useit_opt": True,
                "useit_plot": True,
                "good_user": True,
                "weight": 10.0,
                "exists": True,
            },
            {
                "ix_d1": 2,
                "data_type": "alpha.a",
                "merit_type": "target",
                "ele_ref_name": "",
                "ele_start_name": "",
                "ele_name": "END",
                "meas_value": -1.0,
                "model_value": 0.0,
                "design_value": 0.0,
                "useit_opt": True,
                "useit_plot": True,
                "good_user": True,
                "weight": 100.0,
                "exists": True,
            },
        ]
    }
    return tao


# ---- construction / extraction -----------------------------------------


def test_problem_extracts_only_active_variables():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    assert p.n_var == 2
    assert [v.name for v in p.variables] == ["quad[1]", "quad[2]"]
    assert p.variables[0].ele_name == "Q1"
    assert p.variables[1].ele_name == "Q2"


def test_problem_extracts_only_active_datums():
    tao = _simple_problem_tao()
    tao.d_arrays[("twiss", "end")][1]["useit_opt"] = False
    p = TaoOptimizationProblem(tao)
    assert p.n_data == 1
    assert p.datums[0].name == "twiss.end[1]"


def test_unbounded_sentinels_become_inf():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    # quad[2] had ±1e30 limits — must translate to ±inf.
    assert math.isinf(p.variables[1].low_lim) and p.variables[1].low_lim < 0
    assert math.isinf(p.variables[1].high_lim) and p.variables[1].high_lim > 0


def test_finite_limits_preserved():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    assert p.variables[0].low_lim == -5.0
    assert p.variables[0].high_lim == 5.0


def test_x0_matches_model_values():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    np.testing.assert_array_equal(p.x0, np.array([0.5, -0.3]))


def test_bounds_and_bounds_array_shapes():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    assert p.bounds == [(-5.0, 5.0), (-math.inf, math.inf)]
    lb, ub = p.bounds_array
    assert lb.shape == (2,) and ub.shape == (2,)
    np.testing.assert_array_equal(lb, [-5.0, -math.inf])
    np.testing.assert_array_equal(ub, [5.0, math.inf])


def test_weights_vector():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    np.testing.assert_array_equal(p.weights, [10.0, 100.0])


def test_variable_names_order_stable():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    assert p.variable_names == ["quad[1]", "quad[2]"]


def test_warns_when_no_active_variables(caplog):
    tao = _simple_problem_tao()
    for row in tao.var_v_array_rows["quad"]:
        row["useit_opt"] = False
    with caplog.at_level("WARNING", logger="pytao.optimize.problem"):
        TaoOptimizationProblem(tao)
    assert any("no active variables" in rec.message for rec in caplog.records)


def test_warns_when_no_active_datums(caplog):
    tao = _simple_problem_tao()
    for row in tao.d_arrays[("twiss", "end")]:
        row["useit_opt"] = False
    with caplog.at_level("WARNING", logger="pytao.optimize.problem"):
        TaoOptimizationProblem(tao)
    assert any("no active datums" in rec.message for rec in caplog.records)


# ---- set_variables ------------------------------------------------------


def test_set_variables_emits_one_set_var_per_entry():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    p.set_variables(np.array([1.5, -0.8]))
    assert len(tao.commands) == 2
    assert tao.commands[0].startswith("set var quad[1]|model = ")
    assert tao.commands[1].startswith("set var quad[2]|model = ")
    # Round-trip through float so we don't depend on the exact repr used.
    parsed = [float(c.rsplit("=", 1)[1]) for c in tao.commands]
    np.testing.assert_allclose(parsed, [1.5, -0.8])


def test_set_variables_shape_mismatch_raises():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    with pytest.raises(ValueError, match="expected shape"):
        p.set_variables(np.array([1.0, 2.0, 3.0]))


def test_reset_restores_initial_values():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem(tao)
    p.set_variables(np.array([2.0, 2.0]))
    tao.commands.clear()
    p.reset()
    assert len(tao.commands) == 2
    parsed = [float(c.rsplit("=", 1)[1]) for c in tao.commands]
    np.testing.assert_allclose(parsed, [0.5, -0.3])


# ---- evaluate_merit -----------------------------------------------------


def test_evaluate_merit_sets_then_reads():
    tao = _simple_problem_tao()
    tao.merit_fn = lambda t: 42.0
    p = TaoOptimizationProblem(tao)
    result = p.evaluate_merit(np.array([1.0, 2.0]))
    assert result == 42.0
    assert any("set var quad[1]|model = 1" in c for c in tao.commands)


# ---- delta / residual computation ---------------------------------------


@pytest.mark.parametrize(
    "merit_type,model,meas,expected",
    [
        ("target", 5.0, 3.0, 2.0),
        ("target", 1.0, 3.0, -2.0),
        ("min", 2.0, 5.0, -3.0),  # below -> penalty
        ("min", 7.0, 5.0, 0.0),  # above -> satisfied
        ("max", 7.0, 5.0, 2.0),  # above -> penalty
        ("max", 2.0, 5.0, 0.0),  # below -> satisfied
        ("abs_min", 2.0, 5.0, -3.0),  # |2|<5 -> penalty
        ("abs_min", 7.0, 5.0, 0.0),  # |7|>5 -> satisfied
        ("abs_max", -7.0, 5.0, 2.0),  # |-7|>5 -> penalty
        ("abs_max", 3.0, 5.0, 0.0),  # |3|<5 -> satisfied
    ],
)
def test_compute_datum_delta_matches_tao_merit_types(merit_type, model, meas, expected):
    d = DatumInfo(
        name="d.end[1]",
        d2_name="d",
        d1_name="end",
        ix_d1=1,
        data_type="x",
        merit_type=merit_type,
        meas_value=meas,
        model_value=model,
        design_value=0.0,
        weight=1.0,
    )
    assert _compute_datum_delta(d) == pytest.approx(expected)


def test_residuals_sum_of_squares_equals_merit():
    """sum(r_i**2) must equal Tao's merit (by construction of r_i)."""
    tao = _simple_problem_tao()

    # Merit fn mirrors what Tao would compute: w*(model-meas)^2 per target datum.
    def merit_fn(t: FakeTao) -> float:
        total = 0.0
        for d in t.d_arrays[("twiss", "end")]:
            if not d["useit_opt"]:
                continue
            total += d["weight"] * (d["model_value"] - d["meas_value"]) ** 2
        return total

    tao.merit_fn = merit_fn
    p = TaoOptimizationProblem(tao)
    r = p.evaluate_residuals(p.x0)
    assert r.shape == (2,)
    np.testing.assert_allclose(np.sum(r**2), p.evaluate_merit(p.x0))


# ---- jacobian -----------------------------------------------------------


def test_jacobian_passes_through_derivative_matrix():
    tao = _simple_problem_tao()
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    tao.derivative_matrix = {1: expected}
    p = TaoOptimizationProblem(tao)
    np.testing.assert_array_equal(p.jacobian(), expected)


def test_jacobian_raises_on_shape_mismatch():
    tao = _simple_problem_tao()
    tao.derivative_matrix = {1: np.ones((3, 5))}  # wrong shape
    p = TaoOptimizationProblem(tao)
    with pytest.raises(ValueError, match="returned shape"):
        p.jacobian()


def test_jacobian_raises_when_universe_missing():
    tao = _simple_problem_tao()
    tao.derivative_matrix = {2: np.ones((2, 2))}
    p = TaoOptimizationProblem(tao, universe=1)
    with pytest.raises(KeyError, match="Universe 1"):
        p.jacobian()


def test_residual_jacobian_scales_by_sqrt_weight():
    tao = _simple_problem_tao()
    # Identity model-jacobian so residual jac = diag(sqrt(w))
    tao.derivative_matrix = {1: np.array([[1.0, 0.0], [0.0, 1.0]])}
    tao.merit_fn = lambda t: 0.0
    p = TaoOptimizationProblem(tao)
    J = p.residual_jacobian(p.x0)
    assert J.shape == (2, 2)
    # First datum weight = 10 → sqrt(10); second weight = 100 → 10.
    np.testing.assert_allclose(J[0, 0], math.sqrt(10.0))
    np.testing.assert_allclose(J[1, 1], 10.0)


def test_residual_jacobian_zeroes_inactive_bound_rows():
    """For a satisfied min/max constraint the residual jac row must be zero."""
    tao = _simple_problem_tao()
    # Convert the first datum to a satisfied 'max' constraint:
    row = tao.d_arrays[("twiss", "end")][0]
    row["merit_type"] = "max"
    row["meas_value"] = 50.0  # bound; model=10 < 50 → delta=0
    tao.derivative_matrix = {1: np.ones((2, 2))}
    tao.merit_fn = lambda t: 0.0
    p = TaoOptimizationProblem(tao)
    J = p.residual_jacobian(p.x0)
    np.testing.assert_array_equal(J[0], np.zeros(2))


# ---- _finite_limit low-level sanity -----------------------------------


def test_finite_limit_threshold():
    assert _finite_limit(5.0, -1) == 5.0
    assert _finite_limit(-5.0, -1) == -5.0
    assert math.isinf(_finite_limit(-1e30, -1)) and _finite_limit(-1e30, -1) < 0
    assert math.isinf(_finite_limit(1e30, 1)) and _finite_limit(1e30, 1) > 0


# ---- VariableInfo / DatumInfo are dataclasses; trivially check immutability ----


def test_variable_info_is_frozen():
    v = VariableInfo(
        name="q[1]",
        v1_name="q",
        index=1,
        initial_value=0.0,
        low_lim=-1.0,
        high_lim=1.0,
        step=1e-4,
        weight=0.0,
        merit_type="target",
        ele_name="Q1",
        attrib_name="k1",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        v.initial_value = 1.0  # type: ignore[misc]
