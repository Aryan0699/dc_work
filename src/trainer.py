"""
Pre-training and training loops for DynamicMaxISModel.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.config import Config
from src.dynamic_graph import DynamicGraph, EdgeEvent
from src.model import DynamicMaxISModel


# ─── alpha / beta computation ────────────────────────────────────────────────


def compute_alpha_beta(G0: DynamicGraph, cfg: Config) -> Tuple[int, int]:
    """
    Return (α, β) hop-distances used by the model variant.

    BCAS  : α = β = round(γ · diam(G₀)),  minimum 1
    NoCAS : α = 0,  β = diam(G₀)
    """
    diam = G0.approximate_diameter()
    if cfg.variant == "BCAS":
        val = max(1, round(cfg.gamma * diam))
        return val, val
    else:                           # NoCAS
        return 0, diam


# ─── Pre-training ────────────────────────────────────────────────────────────


def pretrain(
    model: DynamicMaxISModel,
    G0: DynamicGraph,
    cfg: Config,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Pre-train the model on G₀ by building G₀ edge-by-edge.

    Returns the best pre-trained node memories (n, memory_dim) [detached].
    Model weights are updated in-place to the best run's state.
    """
    edges = G0.edges()
    n = G0.n
    best_loss = float("inf")
    best_state: Optional[dict] = None
    best_memory: Optional[torch.Tensor] = None

    print(f"[pretrain] G₀ has {len(edges)} edges, diameter ≈ {G0.approximate_diameter()}")
    print(f"[pretrain] {cfg.pretrain_runs} runs × {cfg.pretrain_epochs} epochs")

    for run_idx in range(cfg.pretrain_runs):
        # re-initialise model per run
        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        run_best_loss = float("inf")
        run_best_state = None
        run_best_memory = None

        for epoch in range(cfg.pretrain_epochs):
            optimizer.zero_grad()

            # start from empty memory
            memory = torch.zeros(n, cfg.memory_dim, device=device)

            # build G₀ one edge at a time (random order)
            shuffled_edges = edges.copy()
            random.shuffle(shuffled_edges)

            temp_graph = DynamicGraph(n)
            for edge_idx, (u, v) in enumerate(shuffled_edges):
                temp_graph.add_edge(u, v)
                signals = {u: [1.0, 0.0, 1.0], v: [1.0, 0.0, 1.0]}
                memory = model.update_memories(memory, signals, device=device)

                # optional periodic detach for memory saving
                if cfg.pretrain_detach_interval > 0 and (edge_idx + 1) % cfg.pretrain_detach_interval == 0:
                    memory = memory.detach()

            # compute estimates & loss for ALL nodes on fully-built G₀
            all_nodes = list(range(n))
            estimates = model.compute_estimates(memory, temp_graph, all_nodes, device=device)
            dummy = torch.zeros(n, device=device)
            loss = DynamicMaxISModel.compute_loss(all_nodes, estimates, dummy, temp_graph, cfg.c)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < run_best_loss:
                run_best_loss = loss_val
                run_best_state = copy.deepcopy(model.state_dict())
                run_best_memory = memory.detach().clone()

        print(f"  run {run_idx+1}/{cfg.pretrain_runs}  best-epoch loss = {run_best_loss:.4f}")

        if run_best_loss < best_loss:
            best_loss = run_best_loss
            best_state = run_best_state
            best_memory = run_best_memory

    # restore best overall weights
    assert best_state is not None and best_memory is not None
    model.load_state_dict(best_state)
    print(f"[pretrain] done — best loss = {best_loss:.4f}")
    return best_memory


# ─── Training ────────────────────────────────────────────────────────────────


def train(
    model: DynamicMaxISModel,
    G0: DynamicGraph,
    train_events: List[EdgeEvent],
    eval_events: List[EdgeEvent],
    pretrain_memory: torch.Tensor,
    cfg: Config,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train the model on the training portion of the event stream.

    Returns (best_memory, best_all_estimates) at the end of the best
    epoch's training pass – ready for inference through eval + test.
    Model weights are updated in-place.
    """
    alpha, beta = compute_alpha_beta(G0, cfg)
    n = G0.n
    print(f"[train] variant={cfg.variant}  α={alpha}, β={beta}")
    print(f"[train] {len(train_events)} train events, {len(eval_events)} eval events")
    print(f"[train] {cfg.train_epochs} epochs")

    max_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else len(train_events)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_eval_loss = float("inf")
    best_state = None
    best_memory: Optional[torch.Tensor] = None
    best_estimates: Optional[torch.Tensor] = None

    for epoch in range(cfg.train_epochs):
        # ── training pass ────────────────────────────────────────────
        model.train()
        graph = G0.copy()
        memory = pretrain_memory.clone().to(device)
        all_estimates = torch.full((n,), 0.5, device=device)
        total_loss = 0.0

        steps = min(max_steps, len(train_events))
        pbar = tqdm(range(steps), desc=f"epoch {epoch+1}/{cfg.train_epochs} [train]", leave=False)
        for i in pbar:
            event = train_events[i]
            optimizer.zero_grad()

            loss, memory_new, all_estimates = model.forward_step(
                graph, event, memory, all_estimates, alpha, beta, cfg.c, device,
            )

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            memory = memory_new.detach()
            total_loss += loss.item()

            if (i + 1) % 2000 == 0:
                pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

        avg_train = total_loss / steps
        train_end_memory = memory.clone()
        train_end_estimates = all_estimates.clone()
        train_end_graph_state = graph.copy()

        # ── evaluation pass (no gradient) ────────────────────────────
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for event in eval_events:
                memory, all_estimates = model.inference_step(
                    graph, event, memory, all_estimates, alpha, beta, device,
                )
                # We approximate eval loss for epoch selection
                # (no need to recompute loss — just use running metric)

        # Use training loss for epoch selection (eval IS quality could
        # also be used, but train loss is cheaper)
        if avg_train < best_eval_loss:
            best_eval_loss = avg_train
            best_state = copy.deepcopy(model.state_dict())
            # Save the state at end of TRAINING (before eval events)
            # so we can replay eval + test afterwards
            best_memory = train_end_memory
            best_estimates = train_end_estimates

        print(f"  epoch {epoch+1}  avg-train-loss = {avg_train:.4f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"[train] done — best avg-train-loss = {best_eval_loss:.4f}")
    return best_memory, best_estimates  # type: ignore[return-value]


# ─── Inference (testing) ─────────────────────────────────────────────────────


def run_inference(
    model: DynamicMaxISModel,
    graph: DynamicGraph,
    events: List[EdgeEvent],
    memory: torch.Tensor,
    all_estimates: torch.Tensor,
    alpha: int,
    beta: int,
    checkpoint_interval: int,
    device: str = "cpu",
) -> List[Dict]:
    """
    Run trained model on an event stream (eval + test or just test).
    Every *checkpoint_interval* steps, generate an integral IS and record it.

    Returns a list of checkpoint records:
        {'step': int, 'is_nodes': set, 'is_size': int, 'graph_snapshot': DynamicGraph}
    """
    model.eval()
    results: List[Dict] = []

    # we may also need to pass through events before the test split
    total = len(events)
    with torch.no_grad():
        for i, event in enumerate(tqdm(events, desc="inference", leave=False)):
            memory, all_estimates = model.inference_step(
                graph, event, memory, all_estimates, alpha, beta, device,
            )

            # checkpoint?
            if (i + 1) % checkpoint_interval == 0 or i == total - 1:
                is_nodes = DynamicMaxISModel.round_to_independent_set(all_estimates, graph)
                results.append({
                    "step": i + 1,
                    "is_nodes": is_nodes,
                    "is_size": len(is_nodes),
                    "graph_snapshot": graph.copy(),
                })

    return results
