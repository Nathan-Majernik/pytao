# Python-driven optimization

Tao ships with several built-in optimizers (`lmdif`, `lm`, `de`, `svd`,
`geodesic_lm`), selectable from a `tao.init` file via
`global%optimizer = '...'` or interactively with the `run` command. The
`pytao.optimize` subpackage lets you swap those out for any Python optimizer
library — most obviously `scipy.optimize` — while keeping the rest of your
Tao setup (lattice, variables, datums, weights, constraints) exactly as
declared in your init file.

## Why use it?

- Access the full SciPy / lmfit / NLopt / CMA-ES / Bayesian-optimization
  ecosystem without rewriting your model.
- Plug Tao into higher-level workflows: multi-objective optimization,
  uncertainty quantification, surrogate models, parallel grid searches.
- Use Tao as the forward model for analytical Jacobian-based solvers — the
  package wraps Tao's built-in `derivative()` command so scipy can consume
  the analytic Jacobian for free.

All variable bounds, weights, and constraint-style datums declared in the
init file are honoured automatically.

## Quickstart

```python
from pytao import Tao
from pytao.optimize import TaoOptimizationProblem, run_scipy_minimize

tao = Tao(init_file="tao.init", noplot=True)
problem = TaoOptimizationProblem(tao)

print(f"{problem.n_var} active variables, {problem.n_data} active datums")
print(f"Starting merit: {problem.evaluate_merit(problem.x0):.3e}")

result = run_scipy_minimize(problem, method="L-BFGS-B")

print(f"Final merit: {result.fun:.3e}")
print(f"Solution: {dict(zip(problem.variable_names, result.x))}")
```

The Tao instance is left at the solver's final point, so you can immediately
run Tao commands (`show lattice`, `write`, plotting) and see the optimized
state.

## What gets picked up from the init file

A `TaoOptimizationProblem` reads the live Tao state and collects:

- **Variables** with `useit_opt = True` (i.e., `good_user & good_opt`).
  Bounds come from `low_lim` / `high_lim`; values at or past ±10²⁰ are
  treated as "no limit" and translated to `±inf` for the optimizer. Steps,
  weights, merit types, and element/attribute names are preserved.
- **Datums** with `useit_opt = True`, with their `meas_value`, `weight`,
  and `merit_type`. Target-style datums (`merit_type = 'target'`) become
  standard residuals; limit-style datums (`'min'`, `'max'`, `'abs_min'`,
  `'abs_max'`) enter the merit as soft constraints, matching Tao's own merit
  function exactly.

The [optimization chapter of the Tao manual][tao-opt] explains the semantics
of each `merit_type`; `pytao.optimize` reproduces them faithfully.

[tao-opt]: https://www.classe.cornell.edu/bmad/tao_manual.pdf

## Choosing an adapter

Two adapters ship with `pytao.optimize`:

### `run_scipy_minimize`

Wraps `scipy.optimize.minimize`. Treats Tao's merit as a scalar objective.

```python
from pytao.optimize import run_scipy_minimize

# Generic bounded quasi-Newton — a good default.
result = run_scipy_minimize(
    problem,
    method="L-BFGS-B",
    options={"gtol": 1e-10, "maxiter": 500},
)

# Gradient-free — useful for noisy / non-smooth merits.
result = run_scipy_minimize(problem, method="Nelder-Mead")
```

`scipy.optimize.minimize` accepts any of its documented methods here:
`"L-BFGS-B"`, `"TNC"`, `"SLSQP"`, `"Powell"`, `"trust-constr"`,
`"Nelder-Mead"`, `"COBYLA"`, `"COBYQA"` honour ``problem.bounds``; other
methods are run unbounded (a log message is emitted).

For global optimizers that live outside ``minimize`` — e.g.
[`scipy.optimize.differential_evolution`][de] — call them directly:

```python
from scipy.optimize import differential_evolution

result = differential_evolution(
    problem.evaluate_merit,
    bounds=problem.bounds,
    x0=problem.x0,
)
problem.set_variables(result.x)  # leave Tao at the final point
```

[de]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

Bounded methods automatically receive `problem.bounds`; unbounded methods
get a log message explaining that bounds were dropped.

### `run_scipy_least_squares`

Wraps `scipy.optimize.least_squares`. Treats each datum's weighted delta as
an individual residual, which is what nonlinear least-squares solvers want.
This is the closest analogue to Tao's default `lmdif` optimizer.

```python
from pytao.optimize import run_scipy_least_squares

result = run_scipy_least_squares(problem, method="trf")

# Feed scipy the analytic Jacobian from Tao's derivative() command.
result = run_scipy_least_squares(problem, method="trf", use_jacobian=True)
```

For problems where Tao can compute `derivative()` reliably, `use_jacobian=True`
is typically ~5–10× faster than the default 2-point finite differences.

## Rolling your own optimizer

`TaoOptimizationProblem` exposes all the primitives you need to drive any
optimizer that can consume a Python callable:

```python
problem = TaoOptimizationProblem(tao)

problem.x0                      # initial variable vector (np.ndarray)
problem.bounds                  # list of (low, high) tuples, ±inf allowed
problem.bounds_array            # (lb, ub) arrays for least_squares
problem.weights                 # per-datum merit weights
problem.variables               # list[VariableInfo]  (name, limits, step, …)
problem.datums                  # list[DatumInfo]     (meas, weight, merit_type, …)

problem.evaluate_merit(x)       # scalar merit after applying x
problem.evaluate_residuals(x)   # residual vector, sum(r²) == merit
problem.jacobian(x)             # dModel/dVar matrix via Tao's derivative()
problem.residual_jacobian(x)    # jacobian of evaluate_residuals
problem.set_variables(x)        # push x into Tao without evaluating
problem.reset()                 # restore the values captured at construction
```

## Hard vs. soft constraints

Tao's merit function encodes all constraints as weighted squared deltas —
they're "soft": they contribute to the merit but don't forbid infeasible
points. A limit-type datum (`merit_type = 'max'`) with weight `w = 100`
penalises a violation of 0.1 by `100 · 0.1² = 1.0`.

If you want *hard* constraints — the kind scipy's `trust-constr` accepts via
`NonlinearConstraint` — build them yourself from
`problem.datums`:

```python
from scipy.optimize import NonlinearConstraint, minimize

def max_constraint(x, problem):
    datums = [d for d in problem.datums if d.merit_type == "max"]
    problem.set_variables(x)
    return [d.meas_value - problem.tao.data(
        d.d2_name, d.d1_name, dat_index=str(d.ix_d1)
    )["model_value"] for d in datums]

constraints = NonlinearConstraint(
    fun=lambda x: max_constraint(x, problem),
    lb=0.0,  # model must stay at or below meas_value
    ub=np.inf,
)

result = minimize(
    problem.evaluate_merit,
    x0=problem.x0,
    method="trust-constr",
    bounds=problem.bounds,
    constraints=constraints,
)
```

## Installation

SciPy is an optional dependency — install it explicitly:

```bash
pip install pytao[optimize]
# or
pip install pytao scipy
```

The read-only parts of `TaoOptimizationProblem` (enumeration, bounds, merit
evaluation, residual vectors, Jacobians) work without SciPy. Only the
`run_scipy_*` adapters require it.

## Runnable benchmark

[`docs/examples/optimize.py`](examples/optimize.py) in the pytao source tree
is a standalone script that runs Tao's built-in `lmdif` alongside
`run_scipy_minimize` and `run_scipy_least_squares` (both finite-difference
and with Tao's analytic Jacobian) against the same packaged init file, and
prints a side-by-side comparison of final merit and forward-evaluation
counts. Run it with `python docs/examples/optimize.py` from a clone of the
pytao repo (a working Tao binary is required).

## See also

- [API reference](api/optimize.md)
- Tao manual, chapter on optimization
