"""Exact coalition enumeration for projected FRPR action preferences."""
from math import comb

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import softmax

TOL = 1e-7


def project(q, constraints, margin):
    if not len(constraints):
        return q.copy(), 0., 0., 0
    before = margin - constraints @ q
    if before.max() <= 0:
        return q.copy(), 0., 0., 0

    def objective(z):
        delta = z - q
        return 0.5 * (delta @ delta), delta

    result = minimize(objective, q, jac=True, method="SLSQP", constraints={
        "type": "ineq", "fun": lambda z: constraints @ z - margin, "jac": lambda z: constraints,
    }, options={"ftol": 1e-11, "maxiter": 300})
    slack = constraints @ result.x - margin
    feasible_error = max(0., -float(slack.min()))
    active = constraints[slack <= TOL]
    stationarity = nnls(active.T, result.x - q)[1] if len(active) else np.linalg.norm(result.x - q)
    if not result.success or feasible_error > TOL or stationarity > TOL:
        raise RuntimeError(f"Projection failed ({result.status}): {result.message}, "
                           f"primal={feasible_error}, stationarity={stationarity}")
    return result.x, feasible_error, float(stationarity), int(result.nit)


def coalition_rows(q, legal, pairs, outcomes, margin, temperature):
    """Yield a single solve per coalition and values for both outcomes and all repeats."""
    legal = np.asarray(legal, dtype=int)
    q = np.asarray(q, dtype=float)[legal]
    positions = {a: i for i, a in enumerate(legal)}
    constraints = np.zeros((len(pairs), len(legal)))
    for i, (preferred, dispreferred) in enumerate(pairs):
        constraints[i, positions[preferred]] = 1
        constraints[i, positions[dispreferred]] = -1
    for mask in range(1 << len(pairs)):
        use = [i for i in range(len(pairs)) if mask & (1 << i)]
        matrix = constraints[use]
        z, primal, stationarity, iterations = project(q, matrix, margin)
        probabilities = softmax(z / temperature)
        if not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0):
            raise ValueError("Invalid legal softmax probability mass")
        values = np.einsum("a,ram->rm", probabilities, outcomes)
        yield mask, z, probabilities, values, dict(
            before_violations=int(np.sum(matrix @ q < margin)),
            after_violations=int(np.sum(matrix @ z < margin - TOL)),
            max_violation_before=float(np.maximum(margin - matrix @ q, 0).max(initial=0)),
            projection_residual=primal, stationarity_residual=stationarity, solver_iterations=iterations)


def exact_shapley(values, m):
    if len(values) != 1 << m or not np.isfinite(values).all():
        raise ValueError("Exact Shapley requires every finite coalition value")
    masks = np.arange(1 << m)
    sizes = np.array([bin(mask).count("1") for mask in range(1 << m)])
    weights = np.array([1 / (m * comb(m - 1, k)) for k in range(m)])
    phi = []
    for i in range(m):
        bit = 1 << i
        base = masks[(masks & bit) == 0]
        phi.append(np.einsum("s,s...->...", weights[sizes[base]], values[base | bit] - values[base]))
    phi = np.asarray(phi)
    error = phi.sum(axis=0) - (values[-1] - values[0])
    if not np.allclose(error, 0, atol=1e-9, rtol=0):
        raise ValueError(f"Shapley efficiency mismatch: {error}")
    return phi, error


def interactions(values, m):
    """Shapley pair interaction index over the remaining M-2 players."""
    if m < 2:
        return
    masks = np.arange(1 << m)
    sizes = np.array([bin(mask).count("1") for mask in range(1 << m)])
    weights = np.array([1 / ((m - 1) * comb(m - 2, k)) for k in range(m - 1)])
    for i in range(m):
        for j in range(i + 1, m):
            left, right = 1 << i, 1 << j
            base = masks[(masks & (left | right)) == 0]
            delta = values[base | left | right] - values[base | left] - values[base | right] + values[base]
            yield i, j, np.einsum("s,s...->...", weights[sizes[base]], delta)
