"""
The :class:`TaoOptimizationProblem` class: a snapshot of the optimization
variables/datums in a running Tao instance, exposed as callables suitable for
external optimizers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# Finite sentinel used when Tao reports an unbounded side. Tao stores
# ``low_lim`` / ``high_lim`` as plain Fortran reals and uses very large values
# (±1e30) to mean "no limit". scipy's bounded solvers accept ±inf, so we
# translate anything past this threshold to ±inf.
_UNBOUNDED_THRESHOLD = 1e20


class _TaoLike(Protocol):
    """Structural type covering the bits of the Tao API we use."""

    def cmd(self, cmd: str, raises: bool = True) -> list[str]: ...
    def var_general(self, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def var_v_array(self, v1_var: str, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def var(self, var: str, *, raises: bool = True) -> dict[str, Any]: ...
    def data_d2_array(self, ix_uni: str = "", *, raises: bool = True) -> list[str]: ...
    def data_d1_array(self, d2_datum: str, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def data_d_array(
        self, d2_name: str, d1_name: str, *, ix_uni: str = "", raises: bool = True
    ) -> list[dict[str, Any]]: ...
    def merit(self, *, raises: bool = True) -> float: ...
    def derivative(self, *, raises: bool = True) -> dict[int, np.ndarray]: ...

    # Optional — ``set_variables`` uses cmds() when available but falls back
    # to cmd() in a loop otherwise. Kept out of the Protocol so stubs don't
    # have to implement it.


@dataclass(frozen=True)
class VariableInfo:
    """
    Description of a single Tao optimization variable.

    Attributes
    ----------
    name : str
        Fully-qualified variable name in the form ``"v1_name[ix]"``, suitable for
        use with ``tao.cmd("set var <name>|model = <value>")``.
    v1_name : str
        Name of the v1 variable group (e.g. ``"quad"``).
    index : int
        Zero-based index within ``v1_name``.
    initial_value : float
        ``model_value`` at the time the problem was constructed.
    low_lim : float
        Lower bound; ``-inf`` if Tao reports no limit.
    high_lim : float
        Upper bound; ``+inf`` if Tao reports no limit.
    step : float
        Step size suggested by Tao (used for finite-difference fallbacks).
    weight : float
        Merit weight applied to limit-type variable violations.
    merit_type : str
        ``"target"`` (default), ``"limit"``, etc. — mirrors
        ``tao_var_struct%merit_type``.
    ele_name : str
        Name of the lattice element the variable operates on, if any.
    attrib_name : str
        Attribute being varied (e.g. ``"k1"``).
    global_index : int
        One-based position in Tao's canonical v1 → ix_v1 enumeration
        (including inactive variables). Matches the ``ix_var`` column index
        Tao uses in its ``derivative()`` matrix.
    """

    name: str
    v1_name: str
    index: int
    initial_value: float
    low_lim: float
    high_lim: float
    step: float
    weight: float
    merit_type: str
    ele_name: str
    attrib_name: str
    global_index: int


@dataclass(frozen=True)
class DatumInfo:
    """
    Description of a single Tao datum that contributes to the merit.

    Attributes
    ----------
    name : str
        Fully-qualified datum name: ``"d2_name.d1_name[ix_d1]"``.
    d2_name, d1_name : str
        Datum group names.
    ix_d1 : int
        Index within the d1 array.
    data_type : str
        Underlying observable (e.g. ``"beta.a"``).
    merit_type : str
        Determines how the delta is computed. One of ``"target"``, ``"min"``,
        ``"max"``, ``"abs_min"``, ``"abs_max"``, ``"average"``, ``"rms"``,
        ``"integral"``, ``"max-min"``.
    meas_value, model_value, design_value : float
        Target value, current value, and the design reference.
    weight : float
        Merit weight (applied as ``weight * delta**2``).
    global_index : int
        One-based global datum index within the universe (the ``ix_data`` Tao
        uses to key ``derivative()`` rows). Computed by enumerating datums in
        Tao's canonical d2 → d1 → ix_d1 order.
    """

    name: str
    d2_name: str
    d1_name: str
    ix_d1: int
    data_type: str
    merit_type: str
    meas_value: float
    model_value: float
    design_value: float
    weight: float
    global_index: int


@dataclass
class TaoOptimizationProblem:
    """
    A snapshot of a Tao optimization problem, wired for external solvers.

    Build one of these from a running :class:`~pytao.Tao` instance; it reads
    back the active variables and datums (those with ``useit_opt = True``) and
    exposes callables that external optimizers can drive. The underlying Tao
    instance is shared, not copied: every evaluation mutates Tao's model state.
    Call :meth:`reset` to restore the initial variable values.

    Parameters
    ----------
    tao : Tao
        A live Tao instance. Any object satisfying the :class:`_TaoLike`
        protocol also works, which is how the unit tests exercise the logic
        without a real Tao binary.
    universe : int, default=1
        Universe index to query for datums. Tao allows multi-universe
        optimization but most setups use a single universe.

    Notes
    -----
    Tao's merit function is

    .. math::

        M = \\sum_i w_i \\, \\Delta_i^2 + \\sum_j w_j \\, \\Delta_j^2

    where the first sum is over active datums and the second over active
    variables (variables only contribute when ``merit_type == "limit"``).
    Constraints declared as ``merit_type = 'max'`` / ``'min'`` on datums are
    therefore *soft* — they're folded into the merit via their weight. If you
    want *hard* constraints for an external solver that supports them (e.g.
    scipy's ``NonlinearConstraint``), inspect :attr:`datums` and build them
    yourself from the entries whose ``merit_type`` is not ``"target"``.
    """

    tao: _TaoLike
    universe: int = 1
    variables: list[VariableInfo] = field(init=False)
    datums: list[DatumInfo] = field(init=False)

    def __post_init__(self) -> None:
        self.variables = _collect_active_variables(self.tao)
        self.datums = _collect_active_datums(self.tao, self.universe)
        self._x0 = np.array([v.initial_value for v in self.variables], dtype=float)
        self._derivative_recalc_warned = False
        _validate_weights(self.variables, self.datums)
        if not self.variables:
            logger.warning(
                "TaoOptimizationProblem built with no active variables "
                "(check good_user / good_opt flags)."
            )
        if not self.datums:
            logger.warning(
                "TaoOptimizationProblem built with no active datums "
                "(check good_user / good_opt flags)."
            )

    # ---- static views ----------------------------------------------------

    @property
    def n_var(self) -> int:
        """Number of active optimization variables."""
        return len(self.variables)

    @property
    def n_data(self) -> int:
        """Number of active datums contributing to the merit."""
        return len(self.datums)

    @property
    def x0(self) -> np.ndarray:
        """Initial variable vector captured at construction."""
        return self._x0.copy()

    @property
    def variable_names(self) -> list[str]:
        """Fully-qualified names of the active variables, in vector order."""
        return [v.name for v in self.variables]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Per-variable ``(low, high)`` tuples, using ``±inf`` for unbounded."""
        return [(v.low_lim, v.high_lim) for v in self.variables]

    @property
    def bounds_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Bounds reshaped for ``scipy.optimize.least_squares``.

        Returns
        -------
        (lb, ub) : tuple of ndarray
            Each of shape ``(n_var,)``. Unbounded sides are ``±inf``.
        """
        lb = np.array([v.low_lim for v in self.variables], dtype=float)
        ub = np.array([v.high_lim for v in self.variables], dtype=float)
        return lb, ub

    @property
    def weights(self) -> np.ndarray:
        """Per-datum merit weights, in datum vector order."""
        return np.array([d.weight for d in self.datums], dtype=float)

    # ---- core callables --------------------------------------------------

    def set_variables(self, x: np.ndarray) -> None:
        """
        Push a variable vector into Tao's model.

        Commands are issued via :meth:`~pytao.Tao.cmds` when available, which
        suppresses lattice recalculation between each ``set var`` — a single
        recalculation happens the next time Tao reads the model (e.g. the
        following ``merit()`` call). For a Tao-shaped object that doesn't
        expose ``cmds``, falls back to an un-batched loop.

        Parameters
        ----------
        x : ndarray, shape (n_var,)
            New variable values. Values outside ``[low_lim, high_lim]`` are
            still sent to Tao — bound enforcement is the optimizer's job, not
            ours. Tao itself will clip or raise depending on the variable's
            configuration.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_var,):
            raise ValueError(f"set_variables expected shape ({self.n_var},), got {x.shape}")
        commands = [
            f"set var {var.name}|model = {value:.17g}"
            for var, value in zip(self.variables, x, strict=True)
        ]
        cmds_method = getattr(self.tao, "cmds", None)
        if callable(cmds_method):
            cmds_method(commands, suppress_lattice_calc=True, suppress_plotting=True)
        else:
            for command in commands:
                self.tao.cmd(command)

    def evaluate_merit(self, x: np.ndarray) -> float:
        """
        Set the variables to ``x`` and return Tao's scalar merit.

        This is the callable to hand to :func:`scipy.optimize.minimize`.
        """
        self.set_variables(x)
        return float(self.tao.merit())

    @property
    def limit_variables(self) -> list[VariableInfo]:
        """Active variables whose merit_type is ``'limit'``."""
        return [v for v in self.variables if v.merit_type == "limit"]

    def evaluate_residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Set the variables to ``x`` and return the weighted residual vector.

        Returns
        -------
        r : ndarray, shape (n_data + n_limit_var,)
            Residuals such that ``sum(r**2)`` equals Tao's merit function.
            Datum contributions come first, in :attr:`datums` order, followed
            by contributions from each active variable with
            ``merit_type == 'limit'`` (in :attr:`variables` order).

        Notes
        -----
        For datums: ``r_i = sqrt(w_i) * delta_i`` where ``delta_i`` is defined
        by the datum's ``merit_type`` (see the Tao optimization chapter). For
        ``"target"`` datums this is ``model - meas``; for bound-style merit
        types (``"min"``, ``"max"``, ``"abs_min"``, ``"abs_max"``) the delta
        is zero when the bound is satisfied.

        For variables with ``merit_type == 'limit'``: the delta is the signed
        bound violation (``model - high_lim`` when above, ``model - low_lim``
        when below, else zero). Variables with ``merit_type == 'target'`` do
        not contribute to the merit.
        """
        self.set_variables(x)
        datum_snapshot = _fetch_datum_model_values(self.tao, self.datums, self.universe)
        residuals = np.empty(self.n_data + len(self.limit_variables), dtype=float)
        for i, d in enumerate(datum_snapshot):
            delta = _compute_datum_delta(d)
            residuals[i] = math.copysign(math.sqrt(d.weight) * abs(delta), delta)
        offset = self.n_data
        for j, (var, value) in enumerate(
            zip(self.limit_variables, self._limit_values_from(x), strict=True)
        ):
            delta = _compute_variable_limit_delta(var, value)
            residuals[offset + j] = math.copysign(math.sqrt(var.weight) * abs(delta), delta)
        return residuals

    def jacobian(self, x: np.ndarray | None = None) -> np.ndarray:
        """
        Return ``dModel_dVar`` for the active datums and variables.

        Slices the full derivative matrix returned by :meth:`Tao.derivative`
        (which is sized by the *maximum* global datum / variable index, with
        NaN entries for inactive slots) down to the active subset using the
        ``global_index`` attributes captured at construction.

        Parameters
        ----------
        x : ndarray, optional
            If provided, first call :meth:`set_variables` so the Jacobian is
            evaluated at ``x``.

        Returns
        -------
        J : ndarray, shape (n_data, n_var)
            Derivative matrix for active datums/variables only.
        """
        if x is not None:
            self.set_variables(x)
        self._warn_if_derivative_recalc_off()
        deriv = self.tao.derivative()
        if self.universe not in deriv:
            raise KeyError(
                f"Universe {self.universe} not in derivative() output "
                f"(got {sorted(deriv)}). Did you run `set global "
                f"derivative_recalc = T` before optimizing?"
            )
        full = np.asarray(deriv[self.universe], dtype=float)
        if self.n_data == 0 or self.n_var == 0:
            return np.zeros((self.n_data, self.n_var), dtype=float)

        row_ix = np.fromiter((d.global_index - 1 for d in self.datums), dtype=int)
        col_ix = np.fromiter((v.global_index - 1 for v in self.variables), dtype=int)
        if (row_ix < 0).any() or (col_ix < 0).any():
            raise ValueError(
                "global_index must be >= 1 for every active var/datum; got "
                f"row={row_ix.tolist()} col={col_ix.tolist()}"
            )
        if row_ix.max(initial=-1) >= full.shape[0] or col_ix.max(initial=-1) >= full.shape[1]:
            raise ValueError(
                f"derivative() returned shape {full.shape}, but active "
                f"indices extend to row={row_ix.max():d}+1, col={col_ix.max():d}+1. "
                "The active set likely changed after the problem was built."
            )
        projected = full[row_ix[:, None], col_ix[None, :]]
        if np.isnan(projected).any():
            raise ValueError(
                "derivative() returned NaN for at least one active "
                "(datum, variable) pair. Ensure `set global "
                "derivative_recalc = T` and that the datums are actually "
                "differentiable by Tao."
            )
        return projected

    def residual_jacobian(self, x: np.ndarray | None = None) -> np.ndarray:
        """
        Jacobian of :meth:`evaluate_residuals` w.r.t. the variable vector.

        Shape is ``(n_data + n_limit_var, n_var)``. For ``"target"`` datum
        rows, ``d(r_i)/d(x_j) = sqrt(w_i) * dModel/dVar``; for bound-style
        merit types the row is zero when the bound is satisfied. For
        ``merit_type == 'limit'`` variable rows, ``d(r_i)/d(x_j)`` is
        ``sqrt(w) * sign(delta) * δ_{ij}`` when the variable is violating
        one of its bounds, zero otherwise.
        """
        if x is not None:
            self.set_variables(x)
        J_model = self.jacobian()
        datum_snapshot = _fetch_datum_model_values(self.tao, self.datums, self.universe)
        scale = np.empty(self.n_data, dtype=float)
        for i, d in enumerate(datum_snapshot):
            delta = _compute_datum_delta(d)
            if delta == 0.0 and d.merit_type != "target":
                scale[i] = 0.0
            else:
                sign = 1.0
                if d.merit_type in {"abs_max", "abs_min"} and d.model_value < 0:
                    sign = -1.0
                scale[i] = sign * math.sqrt(d.weight)
        J_data = scale[:, None] * J_model

        if not self.limit_variables:
            return J_data

        # Each limit-type variable row has a single non-zero column: the one
        # corresponding to that variable in x.
        x_current = np.asarray(x if x is not None else self.x0, dtype=float)
        var_ix_in_x = {var.name: k for k, var in enumerate(self.variables)}
        J_var = np.zeros((len(self.limit_variables), self.n_var), dtype=float)
        for j, (var, value) in enumerate(
            zip(self.limit_variables, self._limit_values_from(x_current), strict=True)
        ):
            delta = _compute_variable_limit_delta(var, value)
            if delta == 0.0:
                continue
            sign = 1.0 if delta > 0 else -1.0
            J_var[j, var_ix_in_x[var.name]] = sign * math.sqrt(var.weight)
        return np.vstack([J_data, J_var])

    def _limit_values_from(self, x: np.ndarray | None) -> np.ndarray:
        """Return the current values of limit-type variables given optimizer state."""
        if x is None:
            x = self.x0
        x = np.asarray(x, dtype=float).ravel()
        out = np.empty(len(self.limit_variables), dtype=float)
        for j, var in enumerate(self.limit_variables):
            ix = next(i for i, v in enumerate(self.variables) if v.name == var.name)
            out[j] = x[ix]
        return out

    def _warn_if_derivative_recalc_off(self) -> None:
        """Emit a one-shot warning if Tao's ``derivative_recalc`` is off."""
        if self._derivative_recalc_warned:
            return
        try:
            tao_global = self.tao.cmd("pipe global")  # type: ignore[attr-defined]
        except Exception:
            return
        on = any(
            "derivative_recalc" in line and "T" in line.split(";")[-1].strip().upper()
            for line in tao_global
        )
        if not on:
            logger.warning(
                "Tao's global%%derivative_recalc appears to be False; "
                "`jacobian()` may return stale entries. Consider "
                "`tao.cmd('set global derivative_recalc = T')`."
            )
        self._derivative_recalc_warned = True

    def reset(self) -> None:
        """Restore variables to their values when the problem was constructed."""
        self.set_variables(self._x0)

    # ---- convenience factories ------------------------------------------

    def scipy_minimize_callable(self) -> Callable[[np.ndarray], float]:
        """Return a closure suitable for ``scipy.optimize.minimize``."""
        return self.evaluate_merit

    def scipy_residuals_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a closure suitable for ``scipy.optimize.least_squares``."""
        return self.evaluate_residuals


# ---- internal helpers ----------------------------------------------------


def _finite_limit(value: float, sign: int) -> float:
    """Translate Tao's ±1e30 "no limit" sentinels to ±inf."""
    if abs(value) > _UNBOUNDED_THRESHOLD:
        return math.copysign(math.inf, sign)
    return float(value)


def _compute_datum_delta(d: DatumInfo) -> float:
    """
    Compute the residual delta for a datum, mirroring Tao's merit function.

    Matches the behaviour in ``tao_merit.f90`` for the merit types most often
    seen in init files.
    """
    m = d.model_value
    t = d.meas_value
    merit_type = d.merit_type
    if merit_type == "target":
        return m - t
    if merit_type == "min":
        return min(m - t, 0.0)
    if merit_type == "max":
        return max(m - t, 0.0)
    if merit_type == "abs_min":
        return min(abs(m) - t, 0.0)
    if merit_type == "abs_max":
        return max(abs(m) - t, 0.0)
    # average / rms / integral / max-min are aggregate types whose model_value
    # already contains the aggregated number — treat them like "target".
    return m - t


def _compute_variable_limit_delta(var: VariableInfo, value: float) -> float:
    """
    Signed bound violation for a ``merit_type == 'limit'`` variable.

    Returns ``value - high_lim`` when above the upper limit, ``value - low_lim``
    (negative) when below the lower limit, and ``0`` inside the box. Mirrors
    the contribution Tao adds to its merit function for limit-type variables.
    """
    if value > var.high_lim:
        return value - var.high_lim
    if value < var.low_lim:
        return value - var.low_lim
    return 0.0


def _validate_weights(variables: list[VariableInfo], datums: list[DatumInfo]) -> None:
    """Weights must be non-negative so ``sqrt(weight)`` stays real."""
    for d in datums:
        if d.weight < 0:
            raise ValueError(
                f"Datum {d.name!r} has negative weight {d.weight!r}; "
                "Tao's merit requires non-negative weights."
            )
    for v in variables:
        if v.weight < 0:
            raise ValueError(
                f"Variable {v.name!r} has negative weight {v.weight!r}; "
                "Tao's merit requires non-negative weights."
            )


def _collect_active_variables(tao: _TaoLike) -> list[VariableInfo]:
    """
    Enumerate every ``useit_opt == True`` variable from Tao.

    The ``global_index`` is the one-based position in Tao's canonical
    v1 → ix_v1 enumeration (including inactive variables), matching the
    ``ix_var`` column index in ``derivative()``'s matrix.
    """
    results: list[VariableInfo] = []
    counter = 0
    for v1 in tao.var_general():
        v1_name = v1["name"]
        rows = tao.var_v_array(v1_name)
        rows_sorted = sorted(rows, key=lambda r: int(r["ix_v1"]))
        for row in rows_sorted:
            counter += 1
            if not row.get("useit_opt", False):
                continue
            ix_v1 = int(row["ix_v1"])
            full_name = f"{v1_name}[{ix_v1}]"
            detail = tao.var(full_name)
            low = _finite_limit(float(detail.get("low_lim", -math.inf)), -1)
            high = _finite_limit(float(detail.get("high_lim", math.inf)), 1)
            results.append(
                VariableInfo(
                    name=full_name,
                    v1_name=v1_name,
                    index=ix_v1,
                    initial_value=float(detail.get("model_value", row["model_value"])),
                    low_lim=low,
                    high_lim=high,
                    step=float(detail.get("step", 0.0)),
                    weight=float(detail.get("weight", row.get("weight", 0.0))),
                    merit_type=str(detail.get("merit_type", "target")),
                    ele_name=str(detail.get("ele_name", "")),
                    attrib_name=str(detail.get("attrib_name", row.get("var_attrib_name", ""))),
                    global_index=counter,
                )
            )
    return results


def _collect_active_datums(tao: _TaoLike, universe: int) -> list[DatumInfo]:
    """
    Enumerate every ``useit_opt == True`` datum for ``universe``.

    The ``global_index`` is the one-based position in Tao's canonical
    d2 → d1 → ix_d1 enumeration (including inactive datums), matching the
    ``ix_data`` row index in ``derivative()``'s matrix.
    """
    results: list[DatumInfo] = []
    counter = 0
    for d2 in tao.data_d2_array(str(universe)):
        d2_name = d2 if isinstance(d2, str) else d2.get("name", "")
        if not d2_name:
            continue
        d2_ref = f"{universe}@{d2_name}"
        for d1 in tao.data_d1_array(d2_ref):
            d1_name = d1["name"] if isinstance(d1, dict) else d1
            rows = tao.data_d_array(d2_name, d1_name, ix_uni=str(universe))
            rows_sorted = sorted(rows, key=lambda r: int(r["ix_d1"]))
            for row in rows_sorted:
                counter += 1
                if not row.get("useit_opt", False):
                    continue
                ix_d1 = int(row["ix_d1"])
                results.append(
                    DatumInfo(
                        name=f"{d2_name}.{d1_name}[{ix_d1}]",
                        d2_name=d2_name,
                        d1_name=d1_name,
                        ix_d1=ix_d1,
                        data_type=str(row.get("data_type", "")),
                        merit_type=str(row.get("merit_type", "target")),
                        meas_value=float(row["meas_value"]),
                        model_value=float(row["model_value"]),
                        design_value=float(row["design_value"]),
                        weight=float(row.get("weight", 0.0)),
                        global_index=counter,
                    )
                )
    return results


def _fetch_datum_model_values(
    tao: _TaoLike, datums: list[DatumInfo], universe: int
) -> list[DatumInfo]:
    """
    Return a fresh list of :class:`DatumInfo` with up-to-date ``model_value``.

    Re-queries each d1 array in one go (``data_d_array`` is a list fetch, not a
    per-datum one), so cost is O(n_d1) pipe calls rather than O(n_datum).
    """
    grouped: dict[tuple[str, str], list[DatumInfo]] = {}
    for d in datums:
        grouped.setdefault((d.d2_name, d.d1_name), []).append(d)

    refreshed: dict[str, DatumInfo] = {}
    for (d2, d1), entries in grouped.items():
        rows = {int(r["ix_d1"]): r for r in tao.data_d_array(d2, d1, ix_uni=str(universe))}
        for d in entries:
            row = rows.get(d.ix_d1)
            if row is None:
                refreshed[d.name] = d
                continue
            refreshed[d.name] = DatumInfo(
                name=d.name,
                d2_name=d.d2_name,
                d1_name=d.d1_name,
                ix_d1=d.ix_d1,
                data_type=d.data_type,
                merit_type=d.merit_type,
                meas_value=float(row["meas_value"]),
                model_value=float(row["model_value"]),
                design_value=float(row["design_value"]),
                weight=float(row.get("weight", d.weight)),
                global_index=d.global_index,
            )
    # Preserve the original datum ordering so the residual vector lines up
    # with the problem's declared datum order.
    return [refreshed[d.name] for d in datums]
