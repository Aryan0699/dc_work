"""
DynamicMaxISModel – the unsupervised GNN model for Dynamic MaxIS.

Modules (following the paper):
  1. Event Handling   – produces a signal s_t(v) for nodes near the edge event.
  2. Memory (GRU)     – updates high-dimensional node memory using the signal.
  3. Local Aggregation – maps memories → MaxIS membership probability p ∈ [0,1].
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from src.dynamic_graph import DynamicGraph, EdgeEvent


class DynamicMaxISModel(nn.Module):
    """
    Parameters
    ----------
    n : number of nodes (fixed across the dynamic graph).
    memory_dim : dimensionality of node memory vectors.
    signal_dim : dimensionality of the event signal (default 3).
    hidden_dim : hidden dimensionality for local aggregation.
    mlp_hidden_dim : MLP hidden layer width.
    """

    def __init__(
        self,
        n: int,
        memory_dim: int = 64,
        signal_dim: int = 3,
        hidden_dim: int = 64,
        mlp_hidden_dim: int = 32,
    ):
        super().__init__()
        self.n = n
        self.memory_dim = memory_dim
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim

        # ── Memory module (GRU cell) ────────────────────────────────────
        # input = [old_memory ‖ signal]   →  hidden = old_memory
        self.gru_cell = nn.GRUCell(
            input_size=memory_dim + signal_dim,
            hidden_size=memory_dim,
        )

        # ── Local aggregation module ────────────────────────────────────
        # W1: aggregate neighbour memories
        self.W1 = nn.Linear(memory_dim, hidden_dim, bias=False)
        # W2: combine [aggregated ‖ own_memory ‖ degree]
        self.W2 = nn.Linear(hidden_dim + memory_dim + 1, hidden_dim, bias=False)
        # Step-down MLP → scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Reinitialise all learnable weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRUCell):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    # ── 1. Event handling ────────────────────────────────────────────────

    @staticmethod
    def compute_signals(
        event: EdgeEvent,
        distances: Dict[int, int],
        alpha: int,
    ) -> Dict[int, List[float]]:
        """
        Produce signal s_t(v) = [enc(E_t) ‖ r_t(v)]  for each node in *distances*.

        enc: addition → [1,0],  deletion → [0,1]
        r_t(v) = 1 − 2·dist/α   linearly interpolated into [−1, 1].
        """
        enc = [1.0, 0.0] if event.event_type == 1 else [0.0, 1.0]
        signals: Dict[int, List[float]] = {}
        for node, dist in distances.items():
            r = 1.0 - 2.0 * dist / alpha if alpha > 0 else 1.0
            signals[node] = enc + [r]
        return signals

    # ── 2. Memory module (GRU update) ────────────────────────────────────

    def update_memories(
        self,
        memory: torch.Tensor,
        signals: Dict[int, List[float]],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Update memories for nodes that received a signal.
        Returns a **new** memory tensor (computation-graph–safe).
        """
        if not signals:
            return memory

        node_ids = sorted(signals.keys())
        idx = torch.tensor(node_ids, dtype=torch.long, device=device)
        old_mem = memory[idx]                                        # (K, mem_dim)
        sig = torch.tensor(
            [signals[n] for n in node_ids], dtype=torch.float, device=device
        )                                                            # (K, sig_dim)

        gru_input = torch.cat([old_mem, sig], dim=1)                 # (K, mem_dim+sig_dim)
        new_mem = self.gru_cell(gru_input, old_mem)                  # (K, mem_dim)

        # Clone then overwrite only modified entries
        updated = memory.clone()
        updated[idx] = new_mem
        return updated

    # ── 3. Local aggregation → estimates ─────────────────────────────────

    def compute_estimates(
        self,
        memory: torch.Tensor,
        graph: DynamicGraph,
        beta_nodes: List[int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        For each node in *beta_nodes*, compute membership probability p ∈ [0,1].
        Returns tensor of shape (K,).
        """
        K = len(beta_nodes)
        if K == 0:
            return torch.zeros(0, device=device)

        # aggregate neighbour memories
        agg_list: List[torch.Tensor] = []
        deg_list: List[float] = []
        for v in beta_nodes:
            nbrs = list(graph.neighbors(v))
            if nbrs:
                nbr_idx = torch.tensor(nbrs, dtype=torch.long, device=device)
                agg = memory[nbr_idx].sum(dim=0)                    # (mem_dim,)
            else:
                agg = torch.zeros(self.memory_dim, device=device)
            agg_list.append(agg)
            deg_list.append(float(len(nbrs)))

        agg_tensor = torch.stack(agg_list)                           # (K, mem_dim)
        h_tilde = torch.relu(self.W1(agg_tensor))                   # (K, hidden_dim)

        beta_idx = torch.tensor(beta_nodes, dtype=torch.long, device=device)
        own_mem = memory[beta_idx]                                   # (K, mem_dim)
        deg_tensor = torch.tensor(
            deg_list, dtype=torch.float, device=device
        ).unsqueeze(1)                                               # (K, 1)

        concat = torch.cat([h_tilde, own_mem, deg_tensor], dim=1)   # (K, hid+mem+1)
        h = self.W2(concat)                                          # (K, hidden_dim)
        logits = self.mlp(h).squeeze(-1)                             # (K,)
        return torch.sigmoid(logits)                                 # (K,)

    # ── Loss ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        beta_nodes: List[int],
        beta_estimates: torch.Tensor,
        all_estimates: torch.Tensor,
        graph: DynamicGraph,
        c: float = 3.0,
    ) -> torch.Tensor:
        """
        Cumulative loss  Σ_{v ∈ β-nodes} ℓ_t(v)
        where  ℓ_t(v) = −p_v + (c / 2d_v) · p_v · Σ_{u ∈ N(v)} p_u

        *beta_estimates* has grad;  entries of *all_estimates* outside β are
        treated as constants.
        """
        K = len(beta_nodes)
        if K == 0:
            return torch.tensor(0.0)

        device = beta_estimates.device
        beta_map = {v: i for i, v in enumerate(beta_nodes)}

        nbr_sums: List[torch.Tensor] = []
        degrees: List[float] = []

        for v in beta_nodes:
            nbrs = list(graph.neighbors(v))
            deg = max(len(nbrs), 1)
            degrees.append(float(deg))

            s = torch.tensor(0.0, device=device)
            for u in nbrs:
                if u in beta_map:
                    s = s + beta_estimates[beta_map[u]]
                else:
                    s = s + all_estimates[u]            # detached constant
            nbr_sums.append(s)

        nbr_sum_t = torch.stack(nbr_sums)                           # (K,)
        deg_t = torch.tensor(degrees, dtype=torch.float, device=device)

        loss = (-beta_estimates + (c / (2.0 * deg_t)) * beta_estimates * nbr_sum_t).sum()
        return loss

    # ── Full forward step (one edge event) ───────────────────────────────

    def forward_step(
        self,
        graph: DynamicGraph,
        event: EdgeEvent,
        memory: torch.Tensor,
        all_estimates: torch.Tensor,
        alpha: int,
        beta: int,
        c: float = 3.0,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single edge event **during training** (gradient-tracked).

        1. Apply event to graph (in-place).
        2. Update memories via GRU for α-neighbourhood.
        3. Compute estimates for β-neighbourhood.
        4. Compute loss.

        Returns (loss, new_memory, new_all_estimates).
        *new_memory* retains grad until caller detaches after backward.
        *new_all_estimates* is always detached.
        """
        # 1. apply
        graph.apply_event(event)

        # 2. memory update
        if alpha >= 0:
            alpha_dists = graph.bfs([event.u, event.v], max_hops=alpha)
        else:
            alpha_dists = {}
        signals = self.compute_signals(event, alpha_dists, alpha)
        new_memory = self.update_memories(memory, signals, device=device)

        # 3. estimates
        if beta >= graph.n:
            beta_nodes = list(range(graph.n))
        else:
            beta_nodes = sorted(graph.bfs([event.u, event.v], max_hops=beta).keys())
        beta_est = self.compute_estimates(new_memory, graph, beta_nodes, device=device)

        # 4. loss
        loss = self.compute_loss(beta_nodes, beta_est, all_estimates, graph, c)

        # 5. update stored estimates (detached)
        new_all_estimates = all_estimates.clone()
        if len(beta_nodes) > 0:
            beta_idx = torch.tensor(beta_nodes, dtype=torch.long, device=device)
            new_all_estimates[beta_idx] = beta_est.detach()

        return loss, new_memory, new_all_estimates

    # ── Inference step (no gradient) ─────────────────────────────────────

    @torch.no_grad()
    def inference_step(
        self,
        graph: DynamicGraph,
        event: EdgeEvent,
        memory: torch.Tensor,
        all_estimates: torch.Tensor,
        alpha: int,
        beta: int,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one edge event without gradient.
        Returns (new_memory, new_all_estimates).
        """
        graph.apply_event(event)

        # memory
        alpha_dists = graph.bfs([event.u, event.v], max_hops=alpha) if alpha >= 0 else {}
        signals = self.compute_signals(event, alpha_dists, alpha)
        memory = self.update_memories(memory, signals, device=device)

        # estimates
        if beta >= graph.n:
            beta_nodes = list(range(graph.n))
        else:
            beta_nodes = sorted(graph.bfs([event.u, event.v], max_hops=beta).keys())
        beta_est = self.compute_estimates(memory, graph, beta_nodes, device=device)

        if len(beta_nodes) > 0:
            beta_idx = torch.tensor(beta_nodes, dtype=torch.long, device=device)
            all_estimates[beta_idx] = beta_est

        return memory, all_estimates

    # ── Rounding: relaxed estimates → integral IS ────────────────────────

    @staticmethod
    def round_to_independent_set(
        all_estimates: torch.Tensor,
        graph: DynamicGraph,
    ) -> Set[int]:
        """
        1. Include all v with estimate ≥ 0.5.
        2. Greedily remove nodes with most violations until IS is valid.
        """
        candidate = set()
        est_np = all_estimates.detach().cpu().numpy()
        for v in range(graph.n):
            if est_np[v] >= 0.5:
                candidate.add(v)

        # greedy violation removal
        while True:
            violations: Dict[int, int] = {}
            for v in candidate:
                count = 0
                for u in graph.neighbors(v):
                    if u in candidate:
                        count += 1
                if count > 0:
                    violations[v] = count
            if not violations:
                break
            # remove node with most violations (break ties by lower estimate)
            worst = max(violations, key=lambda v: (violations[v], -est_np[v]))
            candidate.discard(worst)

        return candidate
