"""
Baseline solvers for Maximum Independent Set:
  • ILP-CBC  (exact, via PuLP + open-source CBC solver)
  • Greedy   (fast heuristic)
"""

from __future__ import annotations

import time
from typing import Optional, Set, Tuple

from src.dynamic_graph import DynamicGraph


# ─── ILP via PuLP + CBC ─────────────────────────────────────────────────────


def ilp_mis(
    graph: DynamicGraph,
    time_limit: Optional[float] = None,
) -> Tuple[Set[int], int, float]:
    """
    Solve MIS exactly (or near-optimally with time limit) via ILP.

    max  Σ x_v
    s.t. x_u + x_v ≤ 1   ∀ (u,v) ∈ E
         x_v ∈ {0, 1}

    Returns (is_nodes, is_size, elapsed_seconds).
    """
    import pulp

    n = graph.n
    prob = pulp.LpProblem("MaxIS", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in range(n)]

    # objective
    prob += pulp.lpSum(x)

    # independence constraints
    for u in range(n):
        for v in graph.neighbors(u):
            if u < v:
                prob += x[u] + x[v] <= 1

    # solver
    solver_kwargs = {"msg": 0}
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)

    t0 = time.perf_counter()
    prob.solve(solver)
    elapsed = time.perf_counter() - t0

    is_nodes: Set[int] = set()
    for v in range(n):
        if pulp.value(x[v]) is not None and pulp.value(x[v]) > 0.5:
            is_nodes.add(v)

    return is_nodes, len(is_nodes), elapsed


# ─── Greedy heuristic ───────────────────────────────────────────────────────


def greedy_mis(graph: DynamicGraph) -> Tuple[Set[int], int, float]:
    """
    Deterministic greedy MIS:
      1. Sort nodes by degree (ascending).
      2. Add node if none of its neighbours already selected.

    Returns (is_nodes, is_size, elapsed_seconds).
    """
    t0 = time.perf_counter()
    n = graph.n
    nodes_by_degree = sorted(range(n), key=lambda v: graph.degree(v))

    selected: Set[int] = set()
    blocked: Set[int] = set()

    for v in nodes_by_degree:
        if v not in blocked:
            selected.add(v)
            for u in graph.neighbors(v):
                blocked.add(u)

    elapsed = time.perf_counter() - t0
    return selected, len(selected), elapsed


# ─── Unified interface ───────────────────────────────────────────────────────


def run_baseline(
    graph: DynamicGraph,
    method: str = "ILP",
    time_limit: Optional[float] = None,
) -> Tuple[Set[int], int, float]:
    """
    Run a baseline solver.

    Parameters
    ----------
    method : "ILP" or "Greedy"
    time_limit : only used for ILP (seconds)
    """
    if method == "ILP":
        return ilp_mis(graph, time_limit=time_limit)
    elif method == "Greedy":
        return greedy_mis(graph)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
