"""
Evaluation pipeline: run trained model + baselines on checkpoints,
compute quality ratios, validity, and timing.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.baselines import run_baseline
from src.dynamic_graph import DynamicGraph
from src.utils import verify_independent_set


def evaluate_checkpoints(
    model_checkpoints: List[Dict],
    baseline_method: str = "ILP",
    ilp_time_limit: Optional[float] = None,
) -> Dict:
    """
    For each checkpoint produced by `run_inference`, run the baseline
    on the same graph snapshot and compute metrics.

    Parameters
    ----------
    model_checkpoints : list of dicts from trainer.run_inference
        Each dict has: step, is_nodes, is_size, graph_snapshot
    baseline_method : "ILP" or "Greedy"
    ilp_time_limit : time limit for ILP (seconds)

    Returns
    -------
    dict with aggregate metrics and per-checkpoint details.
    """
    quality_ratios: List[float] = []
    model_sizes: List[int] = []
    baseline_sizes: List[int] = []
    baseline_times: List[float] = []
    validity: List[bool] = []
    details: List[Dict] = []

    print(f"\n[evaluate] {len(model_checkpoints)} checkpoints  baseline={baseline_method}")

    for ckpt in tqdm(model_checkpoints, desc="baseline eval"):
        step = ckpt["step"]
        model_is = ckpt["is_nodes"]
        model_size = ckpt["is_size"]
        graph: DynamicGraph = ckpt["graph_snapshot"]

        # validity
        valid = verify_independent_set(model_is, graph)
        validity.append(valid)

        # baseline
        bl_nodes, bl_size, bl_time = run_baseline(
            graph, method=baseline_method, time_limit=ilp_time_limit
        )
        baseline_sizes.append(bl_size)
        baseline_times.append(bl_time)
        model_sizes.append(model_size)

        # quality ratio  Q = |I_ours| / |I_baseline|
        q = model_size / bl_size if bl_size > 0 else 1.0
        quality_ratios.append(q)

        details.append({
            "step": step,
            "model_size": model_size,
            "baseline_size": bl_size,
            "quality": q,
            "valid": valid,
            "baseline_time": bl_time,
        })

    q_arr = np.array(quality_ratios)
    return {
        "mean_quality": float(q_arr.mean()),
        "std_quality": float(q_arr.std()),
        "mean_baseline_time": float(np.mean(baseline_times)),
        "validity_pct": 100.0 * sum(validity) / len(validity) if validity else 0.0,
        "num_checkpoints": len(model_checkpoints),
        "details": details,
    }


def print_results_table(
    size_label: str,
    n: int,
    T: int,
    baseline_name: str,
    num_checkpoints: int,
    mean_q: float,
    std_q: float,
    model_ms_per_snap: float,
    baseline_ms_per_ckpt: float,
    valid_pct: float,
) -> None:
    """Print a single row of the result table."""
    hdr = (
        f"{'Scale':<8} {'n':>6} {'T':>8} {'Baseline':<16} "
        f"{'Ckpts':>6} {'Mean Q ± std':>16} "
        f"{'Model ms/snap':>14} {'BL ms/ckpt':>12} {'Valid%':>7}"
    )
    row = (
        f"{size_label:<8} {n:>6} {T:>8} {baseline_name:<16} "
        f"{num_checkpoints:>6} {mean_q:>7.4f} ± {std_q:<6.4f} "
        f"{model_ms_per_snap:>14.2f} {baseline_ms_per_ckpt:>12.2f} {valid_pct:>7.1f}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    print(row)
    print("=" * len(hdr))
