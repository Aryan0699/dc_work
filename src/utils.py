"""
Utility helpers.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def verify_independent_set(nodes: set, graph) -> bool:
    """Return True iff *nodes* is a valid independent set in *graph*."""
    for v in nodes:
        for u in graph.neighbors(v):
            if u in nodes:
                return False
    return True
