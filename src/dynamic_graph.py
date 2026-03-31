"""
Dynamic graph data structure, Erdős–Rényi generator, and edge-event stream generator.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# ─── Edge Event ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EdgeEvent:
    """One edge change in a dynamic graph."""
    u: int
    v: int
    event_type: int          # +1 = addition,  -1 = deletion


# ─── Dynamic Graph ───────────────────────────────────────────────────────────


class DynamicGraph:
    """
    Undirected graph with O(1) edge add / remove / lookup,
    O(deg) neighbour iteration, and fast BFS.
    """

    def __init__(self, n: int):
        self.n = n
        self.adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        self._num_edges = 0

    # ── mutators ─────────────────────────────────────────────────────────

    def add_edge(self, u: int, v: int) -> None:
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.adj[v].add(u)
            self._num_edges += 1

    def remove_edge(self, u: int, v: int) -> None:
        if v in self.adj[u]:
            self.adj[u].discard(v)
            self.adj[v].discard(u)
            self._num_edges -= 1

    def apply_event(self, event: EdgeEvent) -> None:
        if event.event_type == 1:
            self.add_edge(event.u, event.v)
        else:
            self.remove_edge(event.u, event.v)

    # ── queries ──────────────────────────────────────────────────────────

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def neighbors(self, v: int) -> Set[int]:
        return self.adj[v]

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    @property
    def num_edges(self) -> int:
        return self._num_edges

    def edges(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    # ── BFS ──────────────────────────────────────────────────────────────

    def bfs(self, sources: List[int], max_hops: int) -> Dict[int, int]:
        """
        Multi-source BFS.  Returns {node: distance} for all nodes within
        *max_hops* of any source.
        """
        distances: Dict[int, int] = {}
        queue: deque = deque()
        for s in sources:
            if s not in distances:
                distances[s] = 0
                queue.append(s)

        while queue:
            u = queue.popleft()
            d = distances[u]
            if d >= max_hops:
                continue
            for v in self.adj[u]:
                if v not in distances:
                    distances[v] = d + 1
                    queue.append(v)
        return distances

    # ── diameter approximation ───────────────────────────────────────────

    def approximate_diameter(self, num_samples: int = 20) -> int:
        """BFS from several random nodes; return max observed distance."""
        if self.n == 0:
            return 0
        nodes = list(range(self.n))
        samples = random.sample(nodes, min(num_samples, self.n))
        max_dist = 0
        for s in samples:
            dists = self.bfs([s], self.n)       # unlimited hops
            if dists:
                max_dist = max(max_dist, max(dists.values()))
        return max(max_dist, 1)

    # ── copy ─────────────────────────────────────────────────────────────

    def copy(self) -> "DynamicGraph":
        g = DynamicGraph(self.n)
        for u in range(self.n):
            g.adj[u] = set(self.adj[u])
        g._num_edges = self._num_edges
        return g

    def __repr__(self) -> str:
        return f"DynamicGraph(n={self.n}, m={self._num_edges})"


# ─── Generators ──────────────────────────────────────────────────────────────


def generate_er_graph(n: int, p: float, seed: int = 42) -> DynamicGraph:
    """Generate an Erdős–Rényi G(n, p) random graph."""
    rng = random.Random(seed)
    g = DynamicGraph(n)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                g.add_edge(u, v)
    return g


def generate_dynamic_events(
    n: int,
    initial_edges: List[Tuple[int, int]],
    T: int,
    seed: int = 43,
) -> List[EdgeEvent]:
    """
    Generate *T* random edge events (add / delete with prob 0.5 each)
    starting from *initial_edges*.  The caller's graph is NOT modified.

    Parameters
    ----------
    n : number of nodes
    initial_edges : edges of G₀  (list of (u,v) with u < v)
    T : number of events to generate
    seed : random seed
    """
    rng = random.Random(seed)

    # current edge tracking
    edges_set: Set[Tuple[int, int]] = set(initial_edges)
    edges_list: List[Tuple[int, int]] = list(edges_set)
    edge_to_idx: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(edges_list)}

    max_edges = n * (n - 1) // 2
    events: List[EdgeEvent] = []

    for _ in range(T):
        num_edges = len(edges_list)
        num_non_edges = max_edges - num_edges

        # decide add or delete
        if num_edges == 0:
            add = True
        elif num_non_edges == 0:
            add = False
        else:
            add = rng.random() < 0.5

        if add:
            # rejection-sample a random non-edge
            while True:
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in edges_set:
                    edges_set.add((a, b))
                    edges_list.append((a, b))
                    edge_to_idx[(a, b)] = len(edges_list) - 1
                    events.append(EdgeEvent(a, b, 1))
                    break
        else:
            # O(1) random delete via swap-and-pop
            idx = rng.randint(0, len(edges_list) - 1)
            a, b = edges_list[idx]

            last = edges_list[-1]
            edges_list[idx] = last
            edge_to_idx[last] = idx
            edges_list.pop()
            del edge_to_idx[(a, b)]
            edges_set.remove((a, b))

            events.append(EdgeEvent(a, b, -1))

    return events
