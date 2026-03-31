#!/usr/bin/env python3
"""
Large-Scale Experiment Runner (50K, 100K, 500K nodes)
Tests generalization and scalability of the Dynamic MaxIS model.

Usage:
  python run_large_experiments.py [--size {50k,100k,500k,all}] [--device {gpu,cpu,auto}] [--fast]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

from src.config import Config, get_xlarge_50k_config, get_xxlarge_100k_config, get_huge_500k_config
from src.utils import set_seed, ensure_dir
from src.dynamic_graph import generate_er_graph, generate_dynamic_events
from src.model import DynamicMaxISModel
from src.trainer import compute_alpha_beta, pretrain, train, run_inference
from src.baselines import run_baseline
from src.evaluate import evaluate_checkpoints, print_results_table


def resolve_device(device_str: str) -> str:
    """Resolve device preference: 'gpu', 'cuda', 'cpu', or 'auto'."""
    choice = device_str.lower()
    if choice in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            try:
                # Actual test: can we do tensor ops on GPU?
                with torch.no_grad():
                    torch.zeros(1, device="cuda")
                return "cuda"
            except RuntimeError as e:
                print(f"[WARN] CUDA available but test failed: {e}")
                print("[WARN] Falling back to CPU")
                return "cpu"
        else:
            print("[WARN] CUDA not available. Falling back to CPU.")
            return "cpu"
    elif choice == "cpu":
        return "cpu"
    else:  # auto
        return resolve_device("gpu")  # prefer GPU with fallback


def run_single_experiment(cfg: Config, device: str, output_file: Path):
    """Run a single large-scale experiment."""
    
    print(f"\n{'='*80}")
    print(f"Large-Scale Experiment: n={cfg.n:,}, T={cfg.T:,}, p={cfg.p:.6f}")
    print(f"Device: {device}, Pre-train epochs: {cfg.pretrain_epochs}, "
          f"Train epochs: {cfg.train_epochs}")
    print(f"{'='*80}")
    
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    
    # ─── 1. Generate ER graph + events ───────────────────────────────────
    print("\n[1/6] Generating ER graph and event stream...")
    start = time.time()
    G0 = generate_er_graph(cfg.n, cfg.p, seed=cfg.seed)
    print(f"      Graph: n={G0.n}, m={G0.num_edges}, p={cfg.p:.6f}")
    
    events = generate_dynamic_events(cfg.n, list(G0.edges()), cfg.T, seed=cfg.seed)
    print(f"      Event stream: T={len(events)}")
    elapsed = time.time() - start
    print(f"      Time: {elapsed:.2f}s")
    
    # ─── 2. Build model ─────────────────────────────────────────────────
    print("\n[2/6] Building model...")
    start = time.time()
    model = DynamicMaxISModel(
        n=cfg.n,
        memory_dim=cfg.memory_dim,
        signal_dim=cfg.signal_dim,
        hidden_dim=cfg.hidden_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )
    model.to(device)
    elapsed = time.time() - start
    param_count = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {param_count:,}")
    print(f"      Time: {elapsed:.2f}s")
    
    # ─── 3. Compute α, β ────────────────────────────────────────────────
    print("\n[3/6] Computing α, β...")
    start = time.time()
    alpha, beta = compute_alpha_beta(G0, cfg)
    print(f"      α={alpha}, β={beta}, γ={cfg.gamma}")
    elapsed = time.time() - start
    print(f"      Time: {elapsed:.2f}s")
    
    # ─── 4. Pre-training ────────────────────────────────────────────────
    if cfg.pretrain_epochs > 0:
        print(f"\n[4/6] Pre-training ({cfg.pretrain_epochs} epochs, {cfg.pretrain_runs} runs)...")
        start = time.time()
        pretrain_memory = pretrain(model, G0, cfg, device)
        elapsed = time.time() - start
        print(f"      Time: {elapsed:.2f}s ({elapsed/cfg.pretrain_runs/cfg.pretrain_epochs:.2f}s per epoch)")
    else:
        print(f"\n[4/6] Pre-training skipped (generalization mode)")
        pretrain_memory = torch.zeros(cfg.n, cfg.memory_dim, device=device)

    # Build train/eval/test events (for training + replay)
    n_train = int(len(events) * cfg.train_ratio)
    n_eval = int(len(events) * cfg.eval_ratio)
    train_events = events[:n_train]
    eval_events = events[n_train:n_train + n_eval]
    test_events = events[n_train + n_eval:]
    
    # ─── 5. Training ──────────────────────────────────────────────────
    if cfg.train_epochs > 0 and len(train_events) > 0:
        print(f"\n[5/6] Training ({cfg.train_epochs} epochs)...")
        start = time.time()
        final_memory, final_estimates = train(
            model, G0, train_events, eval_events, pretrain_memory, cfg, device
        )
        elapsed = time.time() - start
        print(f"      Time: {elapsed:.2f}s ({elapsed/cfg.train_epochs:.2f}s per epoch)")
    else:
        print(f"\n[5/6] Training skipped (test only)")
        final_memory = pretrain_memory
        final_estimates = torch.full((cfg.n,), 0.5, device=device)
    
    # ─── 6. Inference + Evaluation ─────────────────────────────────────
    print(f"\n[6/6] Running inference on all test events...")
    # Replay train + eval events to reach test start state
    graph = G0.copy()
    memory = pretrain_memory.clone().to(device)
    all_estimates = torch.full((cfg.n,), 0.5, device=device)
    if train_events or eval_events:
        with torch.no_grad():
            for ev in train_events + eval_events:
                memory, all_estimates = model.inference_step(
                    graph, ev, memory, all_estimates, alpha, beta, device
                )

    start = time.time()
    checkpoints = run_inference(
        model, graph, test_events, memory, all_estimates,
        alpha, beta, cfg.checkpoint_interval, device
    )
    elapsed = time.time() - start
    print(f"      Processed {len(test_events)} events in {elapsed:.2f}s "
          f"({elapsed/len(test_events)*1000:.2f}ms per event)")
    print(f"      Checkpoints: {len(checkpoints)}")
    
    # ─── Run baselines and evaluate ──────────────────────────────────────
    print(f"\n[E] Running baselines and evaluating...")
    start = time.time()
    results = evaluate_checkpoints(
        checkpoints,
        baseline_method=cfg.baseline,
        ilp_time_limit=cfg.ilp_time_limit,
    )
    elapsed = time.time() - start
    print(f"      Evaluation time: {elapsed:.2f}s")
    
    # ─── Print results ──────────────────────────────────────────────────
    print_results_table(cfg.n, cfg.T, cfg.variant, cfg.baseline, results)
    
    # ─── Save detailed results ──────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"=== Large-Scale Experiment Results ===\n")
        f.write(f"Config: n={cfg.n}, T={cfg.T}, p={cfg.p}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Variant: {cfg.variant}, Baseline: {cfg.baseline}\n\n")
        
        f.write(f"Mean Q: {results['mean_q']:.6f}\n")
        f.write(f"Std Q: {results['std_q']:.6f}\n")
        f.write(f"Min Q: {results['min_q']:.6f}\n")
        f.write(f"Max Q: {results['max_q']:.6f}\n")
        f.write(f"Valid: {results['valid_pct']:.1f}%\n")
        f.write(f"Model time/step: {results['model_time_ms']:.2f}ms\n")
        f.write(f"Baseline time/ckpt: {results['baseline_time_ms']:.2f}ms\n")
    
    print(f"\n✓ Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run large-scale experiments (50K, 100K, 500K)"
    )
    parser.add_argument(
        "--size", choices=["50k", "100k", "500k", "all"], default="all",
        help="Which experiment(s) to run"
    )
    parser.add_argument(
        "--variant", choices=["BCAS", "NoCAS"], default="BCAS",
        help="Model variant (BCAS or NoCAS)"
    )
    parser.add_argument(
        "--device", choices=["gpu", "cuda", "cpu", "auto"], default="auto",
        help="Device to use (auto prefers GPU with CPU fallback)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: reduce T and pre-train epochs for quick feedback"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Resolve device
    device = resolve_device(args.device)
    print(f"\n[Device] Using: {device.upper()}\n")
    
    # Determine which experiments to run
    experiments = {
        "50k": (get_xlarge_50k_config, "results/results_large_50k.txt"),
        "100k": (get_xxlarge_100k_config, "results/results_large_100k.txt"),
        "500k": (get_huge_500k_config, "results/results_large_500k.txt"),
    }
    
    if args.size != "all":
        experiments = {args.size: experiments[args.size]}
    
    all_results = {}
    
    for size_name, (config_fn, output_file) in experiments.items():
        cfg = config_fn(variant=args.variant, seed=args.seed, device=device)
        
        # Fast mode: reduce complexity
        if args.fast:
            cfg.T = min(1000, cfg.T)
            cfg.pretrain_epochs = max(1, cfg.pretrain_epochs // 5)
            cfg.checkpoint_interval = max(10, cfg.checkpoint_interval * 5)
        
        try:
            results = run_single_experiment(cfg, device, Path(output_file))
            all_results[size_name] = results
        except Exception as e:
            print(f"\n✗ Experiment {size_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY: Large-Scale Experiments")
        print(f"{'='*80}")
        for size_name, results in all_results.items():
            print(f"{size_name}: Q={results['mean_q']:.4f}±{results['std_q']:.4f}, "
                  f"Valid={results['valid_pct']:.0f}%, "
                  f"Model={results['model_time_ms']:.2f}ms/step")


if __name__ == "__main__":
    main()
