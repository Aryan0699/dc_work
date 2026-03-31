"""
Experiment configuration and hyperparameters.
Based on: "Finding Maximum Independent Sets in Dynamic Graphs using Unsupervised Learning"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ─── Graph parameters ───────────────────────────────────────────────
    n: int = 100                    # Number of nodes
    T: int = 50_000                 # Number of time steps (edge events)
    p: float = 0.1                  # ER edge probability  (expected degree ≈ n*p ≈ 10)

    # ─── Model architecture ─────────────────────────────────────────────
    memory_dim: int = 64            # Node memory vector dimension
    signal_dim: int = 3             # 2 (event encoding) + 1 (normalised distance)
    hidden_dim: int = 64            # Hidden dimension after aggregation
    mlp_hidden_dim: int = 32        # Step-down MLP hidden dim
    c: float = 3.0                  # Loss balance hyperparameter

    # ─── Model variant ──────────────────────────────────────────────────
    variant: str = "BCAS"           # "BCAS" or "NoCAS"
    gamma: float = 0.25             # BCAS: α = β = γ · diam(G₀)

    # ─── Training hyper-parameters ──────────────────────────────────────
    lr: float = 1e-3
    pretrain_epochs: int = 20
    pretrain_runs: int = 3
    train_epochs: int = 5
    pretrain_detach_interval: int = 0   # 0 = keep full graph; >0 = detach every K edges
    max_train_steps: int = 0            # 0 = use all training events

    # ─── Data splits (chronological) ────────────────────────────────────
    train_ratio: float = 0.70
    eval_ratio: float = 0.15
    test_ratio: float = 0.15

    # ─── Baseline ───────────────────────────────────────────────────────
    baseline: str = "ILP"               # "ILP" or "Greedy"
    ilp_time_limit: Optional[float] = None   # seconds (None = no limit)
    checkpoint_interval: int = 100       # evaluate baseline every N test steps

    # ─── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results"


# ── Factory helpers ──────────────────────────────────────────────────────


def get_small_config(**overrides) -> Config:
    cfg = Config(
        n=100, T=50_000, p=0.1,
        train_ratio=0.70, eval_ratio=0.15, test_ratio=0.15,
        baseline="ILP", ilp_time_limit=None, checkpoint_interval=100,
        pretrain_epochs=20, train_epochs=5, pretrain_runs=3,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_medium_config(**overrides) -> Config:
    cfg = Config(
        n=1_000, T=100_000, p=0.01,
        train_ratio=0.50, eval_ratio=0.25, test_ratio=0.25,
        baseline="ILP", ilp_time_limit=10.0, checkpoint_interval=100,
        pretrain_epochs=10, train_epochs=3, pretrain_runs=2,
        pretrain_detach_interval=100,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_large_config(**overrides) -> Config:
    cfg = Config(
        n=10_000, T=5_000, p=0.001,
        train_ratio=0.0, eval_ratio=0.0, test_ratio=1.0,
        baseline="ILP", ilp_time_limit=30.0, checkpoint_interval=5,
        pretrain_epochs=0, train_epochs=0,      # generalization: use model trained on small
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_xlarge_50k_config(**overrides) -> Config:
    """50K nodes — test generalization without pre-training"""
    cfg = Config(
        n=50_000, T=25_000, p=0.0002,  # p = 10/n = 10/50000
        train_ratio=0.0, eval_ratio=0.0, test_ratio=1.0,
        baseline="Greedy", ilp_time_limit=None, checkpoint_interval=10,
        pretrain_epochs=0, train_epochs=0,      # skip pre-training
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_xxlarge_100k_config(**overrides) -> Config:
    """100K nodes — pre-train then test"""
    cfg = Config(
        n=100_000, T=50_000, p=0.0001,  # p = 10/n = 10/100000
        train_ratio=0.0, eval_ratio=0.0, test_ratio=1.0,
        baseline="Greedy", ilp_time_limit=None, checkpoint_interval=20,
        pretrain_epochs=5, train_epochs=0, pretrain_runs=1,  # reduced pre-training
        pretrain_detach_interval=500,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_huge_500k_config(**overrides) -> Config:
    """500K nodes — pre-train then test (challenging scalability test)"""
    cfg = Config(
        n=500_000, T=250_000, p=0.00002,  # p = 10/n = 10/500000
        train_ratio=0.0, eval_ratio=0.0, test_ratio=1.0,
        baseline="Greedy", ilp_time_limit=None, checkpoint_interval=50,
        pretrain_epochs=3, train_epochs=0, pretrain_runs=1,  # minimal pre-training
        pretrain_detach_interval=1000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
