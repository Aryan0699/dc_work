#!/bin/bash
# ─── GPU Server Setup & Run Script (SERVER ONLY) ──────────────────────────
# Upload the entire DC_WORK folder to the server, then run:
#   ssh dc_gr2@10.6.0.86
#   cd ~/DC_WORK && chmod +x run_gpu.sh && ./run_gpu.sh
# ──────────────────────────────────────────────────────────────────────────

set -e

echo "=============================="
echo "  Dynamic MaxIS – GPU Setup"
echo "  (SERVER ONLY - TITAN X Pascal)"
echo "=============================="

# ──────────────────────────────────────────────────────────────
# 1. PyTorch Version Strategy
# ──────────────────────────────────────────────────────────────
# TITAN X Pascal = sm_61 (doesn't support PyTorch 2.9 CUDA)
# Solution: Use PyTorch 1.13.1 which supports sm_61
# ──────────────────────────────────────────────────────────────

echo "[1] Installing PyTorch 1.13.1 (compatible with sm_61)..."
pip uninstall torch -y --quiet 2>/dev/null || true
pip install torch==1.13.1 numpy pulp tqdm --quiet

echo ""
echo "[2] Checking PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}')"

if torch.cuda.is_available() 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

# ──────────────────────────────────────────────────────────────
# 3. Detect working device (GPU with fallback)
# ──────────────────────────────────────────────────────────────

DEVICE="cpu"
echo ""
echo "[3] Testing GPU compatibility..."
if python -c "import torch; t=torch.zeros(1,device='cuda'); print('GPU test: SUCCESS'); exit(0)" 2>/dev/null; then
    DEVICE="cuda"
    echo "    ✓ GPU is working! Using CUDA (PyTorch 1.13.1 with sm_61 support)"
else
    DEVICE="cpu"
    echo "    ℹ GPU not available or test failed. Using CPU"
    echo "    (This is OK—results will be correct, just slower)"
fi

echo ""
echo "======================================================================"
echo "  BASELINE EXPERIMENTS (already tested, for reference)"
echo "======================================================================"

echo ""
echo "=============================================="
echo "  SMALL (n=100, T=50k) – BCAS + ILP baseline"
echo "=============================================="
python run_experiment.py --size small --variant BCAS --device $DEVICE 2>&1 | tee results/log_small_BCAS.txt

echo ""
echo "==============================================="
echo "  MEDIUM (n=1000, T=100k) – BCAS + ILP baseline"
echo "==============================================="
python run_experiment.py --size medium --variant BCAS --baseline ILP --device $DEVICE 2>&1 | tee results/log_medium_BCAS.txt

echo ""
echo "=============================================="
echo "  LARGE (n=10000, T=5k) – BCAS + Greedy baseline"
echo "=============================================="
python run_experiment.py --size large --variant BCAS --baseline Greedy --device $DEVICE 2>&1 | tee results/log_large_BCAS.txt

echo ""
echo "======================================================================"
echo "  LARGE-SCALE EXPERIMENTS (NEW: 50K, 100K, 500K)"
echo "======================================================================"

echo ""
echo "==============================================================>"
echo "  XLARGE (n=50K, T=25k) – skip pre-train, test generalization"
echo "==============================================================>"
python run_large_experiments.py --size 50k --device auto 2>&1 | tee results/log_xlarge_50k.txt

echo ""
echo "======================================================================="
echo "  XXLARGE (n=100K, T=50k) – minimal pre-train (5 epochs, 1 run) + test"
echo "======================================================================="
python run_large_experiments.py --size 100k --device auto 2>&1 | tee results/log_xxlarge_100k.txt

echo ""
echo "==============================================================================="
echo "  HUGE (n=500K, T=250k) – minimal pre-train (3 epochs, 1 run) + test [SLOW!]"
echo "==============================================================================="
python run_large_experiments.py --size 500k --device auto 2>&1 | tee results/log_huge_500k.txt

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
echo "Results saved in results/ directory:"
ls -lh results/*.txt | tail -20
