#!/usr/bin/env python
"""
run_experiment.py – end-to-end runner for the Dynamic MaxIS experiments.

Usage examples
--------------
  python run_experiment.py --size small --variant BCAS
  python run_experiment.py --size medium --variant NoCAS
  python run_experiment.py --size small --variant BCAS --fast   # quick sanity run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

# make sure `src` is importable when run from DC_WORK/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, get_large_config, get_medium_config, get_small_config
from src.dynamic_graph import (
    DynamicGraph,
    EdgeEvent,
    generate_dynamic_events,
    generate_er_graph,
)
from src.evaluate import evaluate_checkpoints, print_results_table
from src.model import DynamicMaxISModel
from src.trainer import compute_alpha_beta, pretrain, run_inference, train
from src.utils import ensure_dir, set_seed


# ─── CLI ─────────────────────────────────────────────────────────────────────


def resolve_device(device_str: str) -> str:
    """Resolve device preference: 'gpu', 'cuda', 'cpu', or 'auto'."""
    if device_str.lower() in ["gpu", "cuda", "auto"]:
        if torch.cuda.is_available():
            try:
                # Actual test: can we do tensor ops on GPU?
                with torch.no_grad():
                    torch.zeros(1, device='cuda')
                print("[Device] CUDA available and working ✓")
                return "cuda"
            except RuntimeError as e:
                print(f"[Device] CUDA available but test failed: {e}")
                print("[Device] Falling back to CPU")
                return "cpu"
        else:
            print("[Device] CUDA not available. Using CPU.")
            return "cpu"
    else:
        return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic MaxIS experiment runner")
    p.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size tier (default: small)",
    )
    p.add_argument(
        "--variant",
        choices=["BCAS", "NoCAS"],
        default="BCAS",
        help="Model variant (default: BCAS)",
    )
    p.add_argument(
        "--baseline",
        choices=["ILP", "Greedy"],
        default=None,
        help="Override baseline method (default: ILP for small, ILP with time-limit for medium/large)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast sanity-check mode: fewer events, epochs, checkpoints",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="auto",
        help="torch device (cpu / cuda / auto, prefers GPU with fallback)",
    )
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # resolve device with smart CUDA testing
    args.device = resolve_device(args.device)

    # ── configuration ────────────────────────────────────────────────────
    fast = args.fast
    if args.size == "small":
        cfg = get_small_config(variant=args.variant, seed=args.seed, device=args.device)
        if fast:
            cfg.T = 5_000
            cfg.pretrain_epochs = 5
            cfg.pretrain_runs = 1
            cfg.train_epochs = 2
            cfg.checkpoint_interval = 50
    elif args.size == "medium":
        cfg = get_medium_config(variant=args.variant, seed=args.seed, device=args.device)
        if fast:
            cfg.T = 10_000
            cfg.pretrain_epochs = 3
            cfg.pretrain_runs = 1
            cfg.train_epochs = 2
            cfg.checkpoint_interval = 100
    else:  # large
        cfg = get_large_config(variant=args.variant, seed=args.seed, device=args.device)
        if fast:
            cfg.T = 1_000
            cfg.checkpoint_interval = 10

    if args.baseline is not None:
        cfg.baseline = args.baseline

    device = cfg.device
    ensure_dir(cfg.output_dir)

    print("=" * 60)
    print(f" Dynamic MaxIS Experiment  |  size={args.size}  variant={cfg.variant}")
    print(f" n={cfg.n}  T={cfg.T}  p={cfg.p}  baseline={cfg.baseline}")
    print(f" device={device}  seed={cfg.seed}")
    print("=" * 60)

    # ── 1. Generate dataset ──────────────────────────────────────────────
    print("\n[1/5] Generating ER dynamic graph …")
    t0 = time.perf_counter()
    G0 = generate_er_graph(cfg.n, cfg.p, seed=cfg.seed)
    initial_edges = G0.edges()
    events = generate_dynamic_events(cfg.n, initial_edges, cfg.T, seed=cfg.seed + 1)
    t_gen = time.perf_counter() - t0
    print(f"  G₀: {G0}  |  {len(events)} events generated in {t_gen:.2f}s")

    # chronological splits
    T = len(events)
    t_train = int(T * cfg.train_ratio)
    t_eval = int(T * (cfg.train_ratio + cfg.eval_ratio))
    train_events = events[:t_train]
    eval_events = events[t_train:t_eval]
    test_events = events[t_eval:]
    print(f"  split: train={len(train_events)}  eval={len(eval_events)}  test={len(test_events)}")

    # ── 2. Build model ───────────────────────────────────────────────────
    print("\n[2/5] Building model …")
    model = DynamicMaxISModel(
        n=cfg.n,
        memory_dim=cfg.memory_dim,
        signal_dim=cfg.signal_dim,
        hidden_dim=cfg.hidden_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")

    # ── 3. Pre-train ──────────────────────────────────────────────────────
    if cfg.pretrain_epochs > 0:
        print("\n[3/5] Pre-training …")
        pretrain_memory = pretrain(model, G0, cfg, device=device)
    else:
        print("\n[3/5] Skipping pre-training (large / generalisation mode)")
        pretrain_memory = torch.zeros(cfg.n, cfg.memory_dim, device=device)

    # ── 4. Train ──────────────────────────────────────────────────────────
    if cfg.train_epochs > 0 and len(train_events) > 0:
        print("\n[4/5] Training …")
        best_memory, best_estimates = train(
            model, G0, train_events, eval_events, pretrain_memory, cfg, device=device,
        )
    else:
        print("\n[4/5] Skipping training (large / generalisation mode)")
        best_memory = pretrain_memory
        best_estimates = torch.full((cfg.n,), 0.5, device=device)

    # ── 5. Inference + Evaluation ─────────────────────────────────────────
    print("\n[5/5] Inference on test events …")
    alpha, beta = compute_alpha_beta(G0, cfg)

    # Replay from G₀ through train + eval events to reach the test start state
    graph = G0.copy()
    memory = pretrain_memory.clone().to(device)
    all_estimates = torch.full((cfg.n,), 0.5, device=device)

    model.eval()
    print("  replaying train + eval events to reach test start …")
    pre_test_events = train_events + eval_events
    t0 = time.perf_counter()
    with torch.no_grad():
        for ev in tqdm(pre_test_events, desc="replay", leave=False):
            memory, all_estimates = model.inference_step(
                graph, ev, memory, all_estimates, alpha, beta, device,
            )
    replay_time = time.perf_counter() - t0
    print(f"  replayed {len(pre_test_events)} events in {replay_time:.2f}s")

    # Run test events with checkpointing
    t0 = time.perf_counter()
    checkpoints = run_inference(
        model, graph, test_events, memory, all_estimates,
        alpha, beta, cfg.checkpoint_interval, device,
    )
    test_time = time.perf_counter() - t0
    model_ms_per_snap = (test_time / len(test_events) * 1000) if test_events else 0.0
    print(f"  test inference: {len(test_events)} events in {test_time:.2f}s  ({model_ms_per_snap:.2f} ms/snap)")

    # ── Baseline evaluation on checkpoints ────────────────────────────────
    eval_results = evaluate_checkpoints(
        checkpoints,
        baseline_method=cfg.baseline,
        ilp_time_limit=cfg.ilp_time_limit,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print_results_table(
        size_label=args.size.upper(),
        n=cfg.n,
        T=cfg.T,
        baseline_name=f"{cfg.baseline}" + (f"({cfg.ilp_time_limit}s)" if cfg.ilp_time_limit else ""),
        num_checkpoints=eval_results["num_checkpoints"],
        mean_q=eval_results["mean_quality"],
        std_q=eval_results["std_quality"],
        model_ms_per_snap=model_ms_per_snap,
        baseline_ms_per_ckpt=eval_results["mean_baseline_time"] * 1000,
        valid_pct=eval_results["validity_pct"],
    )

    # ── Save checkpoint details ───────────────────────────────────────────
    out_path = os.path.join(cfg.output_dir, f"results_{args.size}_{cfg.variant}.txt")
    with open(out_path, "w") as f:
        f.write(f"size={args.size}  variant={cfg.variant}  n={cfg.n}  T={cfg.T}\n")
        f.write(f"baseline={cfg.baseline}  ilp_time_limit={cfg.ilp_time_limit}\n")
        f.write(f"mean_quality={eval_results['mean_quality']:.4f} ± {eval_results['std_quality']:.4f}\n")
        f.write(f"model_ms_per_snap={model_ms_per_snap:.2f}\n")
        f.write(f"baseline_ms_per_ckpt={eval_results['mean_baseline_time']*1000:.2f}\n")
        f.write(f"validity={eval_results['validity_pct']:.1f}%\n\n")
        f.write(f"{'step':>8} {'model':>8} {'baseline':>8} {'quality':>10} {'valid':>6}\n")
        f.write("-" * 46 + "\n")
        for d in eval_results["details"]:
            f.write(
                f"{d['step']:>8} {d['model_size']:>8} {d['baseline_size']:>8} "
                f"{d['quality']:>10.4f} {str(d['valid']):>6}\n"
            )
    print(f"\nDetailed results saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
