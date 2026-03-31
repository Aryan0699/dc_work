,# Dynamic Maximum Independent Set — Implementation & Results Report

**Paper:** "Finding Maximum Independent Sets in Dynamic Graphs using Unsupervised Learning"  
**Authors:** Devendra Parkar, Anya Chaturvedi, Andréa W. Richa, Joshua J. Daymude  
**Source:** arXiv:2505.13754

---

## 1. What Problem Are We Solving?

### The MaxIS Problem
Given a graph G = (V, E), a **Maximum Independent Set (MaxIS)** is the largest possible set
of nodes such that **no two selected nodes share an edge**.

Example: In a social network, it's the largest group of people where none of them are mutual friends.

This problem is:
- **NP-hard** (no polynomial-time exact algorithm known)
- **Hard to approximate** for general graphs

### Why Dynamic?
Real-world graphs change over time:
- Stock correlations fluctuate → portfolio diversification graph changes
- Train delays → transportation scheduling graph changes
- Brain activity changes → functional brain network changes

A **dynamic graph** here is defined as:
- Fixed set of nodes V (nodes never added/removed)
- A sequence of snapshots: G₀, G₁, G₂, ..., Gₜ
- Exactly ONE edge is added or deleted per time step

The goal: **maintain a good MaxIS solution after each edge change**, without recomputing from scratch.

---

## 2. What Approach Did We Implement?

We implemented the **unsupervised GNN-based learning model** from the paper directly.

The key idea: instead of re-solving from scratch at every time step, the model **learns an update
mechanism** — given an edge event (add/delete), it learns which nodes need to change their
MaxIS membership.

### The Model Has Three Modules

```
Edge Event (u,v added/deleted)
         │
         ▼
┌─────────────────────┐
│  1. Event Handling  │  ── produces a signal for nodes near the edge event
│     Module          │     signal = [event_type | distance_to_event]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Memory Module   │  ── updates each node's internal representation
│     (GRU Cell)      │     memory_new = GRU([old_memory | signal], old_memory)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Local Aggregation│ ── each node computes probability of being in MaxIS
│     Module (GNN)    │     by aggregating neighbour memories + own memory + degree
└─────────────────────┘
         │
         ▼
  p(v) ∈ [0,1]  ← probability that node v is in the MaxIS

  Round → valid integral MaxIS solution
```

### The Two Model Variants

| Variant | α (memory update radius) | β (estimate update radius) | Description |
|---------|--------------------------|----------------------------|-------------|
| **BCAS** (Bounded Cascading) | γ · diam(G₀) | γ · diam(G₀) | Only updates a LOCAL region around the edge event (fast, like rule-based algorithms) |
| **NoCAS** (No Cascading) | 0 (only 2 edge nodes) | full diameter | Only edge-incident nodes update memory, but all nodes update estimates |

### The Loss Function (Unsupervised)
No labels needed. The loss encourages large independent sets while penalizing violations:

    ℓ(v) = −p(v)  +  (c / 2·degree(v)) · p(v) · Σ_{u ∈ neighbors(v)} p(u)

- **−p(v)**: reward nodes for having high membership probability (want large IS)
- **+penalty term**: punish when two adjacent nodes both have high probability (enforce independence)
- **c = 3**: balance hyperparameter (from the paper)

### Pre-Training
Before training on the event stream, the model is pre-trained by:
1. Starting from an empty graph
2. Adding edges of G₀ one by one
3. After G₀ is fully built, computing loss for all nodes
4. This gives the model a warm start — it already has a good approximate MaxIS for G₀

### Rounding: Estimates → Valid Integer Solution
After the model outputs probabilities p(v) ∈ [0,1]:
1. Include all nodes with p(v) ≥ 0.5
2. While any two included nodes share an edge:
   → Remove the node with the most violations (breaking ties by lowest probability)
3. Result is always a **100% valid** independent set

---

## 3. Dataset Setup

### Graph Family: Erdős–Rényi (ER) Graphs
- Simplest and most standard benchmark
- Controllable density: expected degree ≈ 10 → p = 10/n

### Three Size Tiers (matching the paper)

| Size   | Nodes (n) | Time Steps (T) | Edge Probability (p) | Avg Edges in G₀ |
|--------|-----------|----------------|----------------------|-----------------|
| Small  | 100       | 50,000         | 0.10                 | ~474            |
| Medium | 1,000     | 100,000        | 0.01                 | ~4,985          |
| Large  | 10,000    | 5,000          | 0.001                | ~49,722         |

### Dynamic Event Generation
At each time step t:
- With probability 0.5: ADD a random non-edge
- With probability 0.5: DELETE a random existing edge
- (If graph is full → force delete; if empty → force add)

### Data Splits (chronological — no future leakage)

| Size   | Train | Eval  | Test  |
|--------|-------|-------|-------|
| Small  | 70%   | 15%   | 15%   |
| Medium | 50%   | 25%   | 25%   |
| Large  | 0%    | 0%    | 100%  |

Large is **generalization only** — the model trained on small/medium is directly applied without retraining.

---

## 4. Baseline Solvers

### Small → ILP-CBC (Exact Optimum)
- Formulate as Integer Linear Program (ILP):
  - Maximize: Σ x_v
  - Subject to: x_u + x_v ≤ 1 for every edge (u,v)
  - x_v ∈ {0, 1}
- Solved with the free CBC solver via PuLP library
- This gives the **provably optimal** MaxIS → quality ratio 1.0 = perfect

### Medium → ILP-CBC with 10-second time limit
- Exact ILP times out for n=1,000, so a 10-second limit is imposed
- Still produces very strong near-optimal solutions (strong baseline)

### Large → Greedy
- Sort nodes by degree (ascending)
- Greedily add node if no neighbour already selected
- Fast (O(n log n)), deterministic, but sub-optimal

### Checkpointing Strategy
Running the baseline on all 50,000–100,000 snapshots is too expensive, so we use checkpoints:

| Size   | Checkpoint Interval | # Checkpoints |
|--------|---------------------|---------------|
| Small  | every 100 steps     | 500           |
| Medium | every 100 steps     | 250           |
| Large  | every 5 steps       | 1,000         |

Our model runs on **ALL** snapshots. The baseline runs only on checkpoints.

---

## 5. Metrics

### 1. Quality Ratio (Approximation Ratio)
    Q(t) = |I_ours(t)| / |I_baseline(t)|

- Q = 1.0 → our IS is the same size as the baseline
- Q > 1.0 → our IS is **better** than the baseline
- Q < 1.0 → our IS is smaller (e.g., Q=0.90 means 90% of optimal)

We report: **mean Q ± std** over all checkpoints.

### 2. Validity (%)
For every output independent set I, we verify:
    For all v ∈ I: no neighbour of v is also in I

Must be 100% — the rounding procedure guarantees this.

### 3. Runtime
- **Model: ms per snapshot** (time to process one edge event)
- **Baseline: ms per checkpoint** (time to solve MIS on a static snapshot)

---

## 6. Results

### Summary Table

| Scale  | n      | T       | Baseline           | Ckpts | Mean Q ± Std      | Model ms/snap | Baseline ms/ckpt | Valid% |
|--------|--------|---------|-------------------|-------|-------------------|---------------|------------------|--------|
| Small  | 100    | 50,000  | ILP-CBC (Exact)   | 500   | 0.7272 ± 0.0830   | 1.23          | 3,109            | 100%   |
| Medium | 1,000  | 100,000 | ILP-CBC (10s lim) | 250   | 0.9068 ± 0.0360   | 3.86          | 9,926            | 100%   |
| Large  | 10,000 | 5,000   | Greedy            | 1,000 | 1.0150 ± 0.0045   | 11,436        | 6.18             | 100%   |

---

### Small Graph (n=100, T=50,000) — BCAS vs Exact ILP

- **Mean Quality: 0.7272 ± 0.0830**
  → Our model finds about **73% of the provably optimal** MaxIS on average
- **Model speed: 1.23 ms/snap** vs **3,109 ms/checkpoint** for ILP
  → **~2,500× faster** than the exact solver
- **Validity: 100%** — every output is a valid independent set
- Quality drops as the graph evolves more (later test steps → lower quality),
  which makes sense as the model was trained on earlier snapshots

### Medium Graph (n=1,000, T=100,000) — BCAS vs ILP (10s limit)

- **Mean Quality: 0.9068 ± 0.0360**
  → Our model finds **~91% of the ILP solution** (ILP here is already time-limited)
- **Model speed: 3.86 ms/snap** vs **9,926 ms/checkpoint** for ILP
  → **~2,570× faster** than ILP
- **Validity: 100%**
- Very consistent quality (low std of 0.036) — the model generalizes well at medium scale
- Individual checkpoint ratios range from 0.86 to 0.99 — close to ILP solutions

### Large Graph (n=10,000, T=5,000) — BCAS vs Greedy

- **Mean Quality: 1.0150 ± 0.0045**
  → Our model **outperforms** the Greedy baseline on every single checkpoint (Q > 1.0)
- This is the **generalization experiment** — model was trained on small/medium graphs
  and directly applied to 10,000-node graphs (100× larger) without retraining
- Individual results: model finds ~2,810–2,833 node IS vs Greedy's ~2,758–2,767
- **Validity: 100%**
- **Note:** At this scale on CPU, the model runs at ~11.4 seconds/snapshot due to
  Python-level graph operations (BFS over 10k nodes). On a properly matched GPU this
  would be significantly faster.

---

## 7. Key Observations

### 1. Quality improves with graph size
| Scale  | Quality |
|--------|---------|
| Small  | 73%     |
| Medium | 91%     |
| Large  | 101.5% (beats Greedy!) |

This is expected: at small scale the ILP baseline gives the exact optimum (tougher target),
while at large scale the Greedy baseline is sub-optimal (easier target to beat).

### 2. Model always produces valid independent sets (100%)
The rounding + violation removal procedure guarantees correctness every time.

### 3. Runtime advantage over exact solvers is massive
- 2,500–2,570× faster than ILP at small/medium scale
- The model processes one edge event in ~1–4 ms (small/medium)

### 4. Generalization works
The model successfully generalizes from n=100/1,000 training graphs to n=10,000 test graphs,
even outperforming the Greedy baseline consistently.

### 5. The BCAS variant works well
With α = β = 0.25 × diameter(G₀), the model only updates a local region around each
edge event. This is fast and effective — it mirrors rule-based dynamic algorithms.

---

## 8. File Structure

```
DC_WORK/
├── run_experiment.py          ← Main experiment runner (CLI entry point)
├── run_gpu.sh                 ← Server run script (auto-detects GPU/CPU)
├── requirements.txt           ← Python dependencies
├── src/
│   ├── config.py              ← All hyperparameters + factory functions for each size
│   ├── dynamic_graph.py       ← DynamicGraph class + ER generator + event stream generator
│   ├── model.py               ← DynamicMaxISModel (all 3 modules + loss + rounding)
│   ├── trainer.py             ← pretrain() + train() + run_inference() functions
│   ├── baselines.py           ← ILP-CBC (exact) and Greedy MIS solvers
│   ├── evaluate.py            ← Checkpoint evaluation, quality ratio, timing
│   └── utils.py               ← Seeding, directory creation, IS validity check
└── results/
    ├── results_small_BCAS.txt   ← Small n=100 detailed results
    ├── results_medium_BCAS.txt  ← Medium n=1000 detailed results
    └── results_large_BCAS.txt   ← Large n=10000 detailed results
```

---

## 9. How to Run

```bash
# Install dependencies
pip install torch numpy pulp tqdm

# Small graph (n=100, T=50k) — ~15 min on CPU
python run_experiment.py --size small --variant BCAS

# Medium graph (n=1000, T=100k) — ~2-3 hrs on CPU
python run_experiment.py --size medium --variant BCAS --baseline ILP

# Large graph (n=10000, T=5k) — generalization test
python run_experiment.py --size large --variant BCAS --baseline Greedy

# Quick sanity check (~1 min)
python run_experiment.py --size small --variant BCAS --fast
```

---

## 10. Limitations & Future Work

1. **Runtime at large scale (CPU):** At n=10,000 the model takes ~11 seconds per snapshot
   due to Python-level BFS and neighbour-iteration loops. The neural network forward pass
   is small (40k parameters) — the bottleneck is graph traversal, not the GNN itself.
   Optimization (vectorized adjacency matrices, compiled graph ops) would bring this down
   dramatically.

2. **GPU incompatibility:** The available server GPU (TITAN X Pascal, sm_61) is too old for
   PyTorch 2.9 (requires sm_70+). All experiments were run on CPU.

3. **Medium NoCAS and Large NoCAS** experiments are pending — the NoCAS variant updates
   all nodes globally per event which is expensive at large scale without optimization.

4. **Baseline for large:** We used Greedy (which our model beats). A stronger baseline
   like a time-limited ILP or KaMIS would give a fairer comparison at large scale.

5. **Single graph family:** Only Erdős–Rényi graphs were tested. The paper also tests on
   Power Law, Twitter, Germany, and Brain network topologies.
