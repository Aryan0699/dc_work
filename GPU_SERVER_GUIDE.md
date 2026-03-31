# GPU Server — Quick Start Guide (SERVER ONLY)

⚠️ **IMPORTANT: All experiments must run on the SERVER (dc_gr2@10.6.0.86) ONLY**  
Your laptop cannot run these experiments due to computational demands.

---

## GPU Strategy for TITAN X Pascal

Your server has a **TITAN X Pascal** (sm_61) which doesn't support PyTorch 2.9's CUDA requirements.

**SOLUTION:** The `run_gpu.sh` script now:
1. ✅ Installs **PyTorch 1.13.1** (has full sm_61 support)
2. ✅ Tests if GPU works with this version
3. ✅ Uses GPU if available, falls back to CPU if not
4. ✅ Runs all experiments transparently

You don't need to do anything — the script handles it automatically!

---

## Summary of Changes

### ✅ What's Been Done (Baseline Experiments)
- **Small** (n=100): Q=0.7272±0.0830 ✓
- **Medium** (n=1,000): Q=0.9068±0.0360 ✓
- **Large** (n=10,000): Q=1.0150±0.0045 ✓

### 🚀 What's New (Large-Scale Testing)
Three new graph scales added to test **scalability & generalization**:

| Scale | n | T | Pre-train | Baseline |
|---|---|---|---|---|
| **XLARGE** | 50K | 25K | ❌ Skip (generalization) | Greedy |
| **XXLARGE** | 100K | 50K | ✅ Mini (5 epochs) | Greedy |
| **HUGE** | 500K | 250K | ✅ Mini (3 epochs) | Greedy |

---

## Running on GPU Server

### Option 1: Full Automated Run (Recommended)

**On your local machine:**
```powershell
cd c:\Users\Ashok Jain\Desktop\DC_WORK

# Upload all files to server
scp -r "DC_WORK\*" dc_gr2@10.6.0.86:~/DC_WORK/
```

**On the server (SSH in):**
```bash
ssh dc_gr2@10.6.0.86
cd ~/DC_WORK
chmod +x run_gpu.sh
./run_gpu.sh   # Runs ALL experiments (baseline + large-scale)
```

This will:
- Auto-detect if GPU is available
- Run baseline experiments (small/medium/large)
- Run new large-scale experiments (50K/100K/500K)
- Save all results to `results/` directory with detailed logs

---

### Option 2: Individual Control

**Run only large-scale experiments:**
```bash
# 50K nodes (skip pre-train, test only)
python run_large_experiments.py --size 50k --device auto

# 100K nodes (mini pre-train 5 epochs, then test)
python run_large_experiments.py --size 100k --device auto

# 500K nodes (mini pre-train 3 epochs, then test)
python run_large_experiments.py --size 500k --device auto

# All three at once
python run_large_experiments.py --size all --device auto
```

**Run with explicit GPU (fail if GPU not available):**
```bash
python run_large_experiments.py --size 100k --device gpu
```

**Run with CPU override:**
```bash
python run_large_experiments.py --size 100k --device cpu
```

**Fast mode (quick feedback, reduced T/epochs):**
```bash
python run_large_experiments.py --size 50k --device auto --fast
```

---

## Device Handling

The scripts now use **smart GPU detection** with CPU fallback:

1. **Auto mode** (default): Tries your GPU, falls back to CPU if incompatible
2. **GPU-only mode**: `--device gpu` — fails if GPU doesn't work
3. **CPU-only mode**: `--device cpu` — ignores GPU entirely

The **TITAN X Pascal** on your server (sm_61) is not compatible with PyTorch 2.9 (needs sm_70+), so:
- You'll see: ⚠ GPU not compatible... Using CPU fallback
- **This is expected and OK** — code still runs correctly on CPU

---

## Results Location

All results saved to `results/` directory:

```
results/
├── results_large_50k.txt       # Quick summary for 50K
├── results_large_100k.txt      # Quick summary for 100K
├── results_large_500k.txt      # Quick summary for 500K
├── log_xlarge_50k.txt          # Full run log for 50K
├── log_xxlarge_100k.txt        # Full run log for 100K
└── log_huge_500k.txt           # Full run log for 500K (slowest)
```

---

## Estimated Run Times (CPU)

| Scale | Events | Approx Time |
|---|---|---|
| 50K | 25,000 | ~2-3 hours |
| 100K | 50,000 | ~5-8 hours |
| 500K | 250,000 | ~24-48 hours |

These are rough estimates for CPU. If GPU works, expect **10-100× speedup**.

---

## Code Changes Made

### 1. **New Config Functions** (`src/config.py`)
- `get_xlarge_50k_config()` — 50K nodes, no pre-train
- `get_xxlarge_100k_config()` — 100K nodes, mini pre-train (5 epochs)
- `get_huge_500k_config()` — 500K nodes, mini pre-train (3 epochs)

### 2. **New Experiment Runner** (`run_large_experiments.py`)
- Dedicated script for 50K/100K/500K experiments
- Better device detection with actual CUDA tensor test
- Cleaner output formatting

### 3. **Improved Device Handling**
- Both `run_experiment.py` and `run_large_experiments.py` now:
  - Actually test if GPU works (not just `cuda.is_available()`)
  - Fallback to CPU gracefully if GPU test fails
  - Show clear device status at startup

### 4. **Updated GPU Script** (`run_gpu.sh`)
- Now runs ALL 8 experiments (baseline + large-scale)
- Uses same smart device detection

---

## Example Output

```
[Device] CUDA available and working ✓
[1/6] Generating ER graph and event stream...
      Graph: n=50000, m=250000, p=0.0002
      Event stream: T=25000
      Time: 3.12s

[2/6] Building model...
      Parameters: 40,001
      Time: 0.15s

[3/6] Computing α, β...
      α=1, β=1, γ=0.25
      Time: 12.34s

[4/6] Pre-training skipped (generalization mode)

[5/6] Training skipped (test only)

[6/6] Running inference on all test events...
      Processed 25000 events in 7532.15s (301.29ms per event)
      Checkpoints: 2500

[E] Running baselines and evaluating...
      Evaluation time: 123.45s

─────────────────────────────────────
|  Q (Quality) │ 0.9847 ± 0.0125 │
│  Model Time  │ 301.29 ms/step  │
│  Baseline    │ 0.42 ms/ckpt    │
│  Valid       │ 100%            │
─────────────────────────────────────
```

---

## FAQ

**Q: Will GPU actually speed this up?**  
A: Yes! The script now uses **PyTorch 1.13.1** which supports TITAN X Pascal (sm_61). GPU should work automatically with 10–100× speedup. If GPU test fails, script falls back to CPU (still correct).

**Q: How long should I wait?**  
A: With GPU (1.13.1): 50K ~20–30 min, 100K ~1–2 hrs. With CPU fallback: 50K ~2–3 hrs. Start with 50K to test your device.

**Q: My results look wrong?**  
A: If Q (quality ratio) is:
- `< 0.5`: Model is much worse than baseline (needs retraining)
- `0.5–1.0`: Model is reasonable
- `> 1.0`: Model **beats** the baseline! (good generalization)

**Q: Can I cancel and restart?**  
A: Yes, interrupt with Ctrl+C. Results files are incrementally saved, not overwritten.

**Q: Can I run this on my laptop?**  
A: No. Server only. 50K–500K node graphs need 10–100+ GB RAM and continuous processing for hours/days.

**Q: Why PyTorch 1.13.1 instead of 2.9?**  
A: PyTorch 2.9 needs sm_70+. Your TITAN X is sm_61. PyTorch 1.13.1 supports sm_61 and is more widely compatible.

**Q: Why skip pre-training for 50K?**  
A: Tests if the model trained on small (n=100) can generalize to 500× larger graphs without retraining. This is a major research question.

---

## Next Steps

1. Upload files to server
2. Run: `./run_gpu.sh`
3. Monitor progress with: `tail -f results/log_xlarge_50k.txt`
4. Check results in `results/` when done
