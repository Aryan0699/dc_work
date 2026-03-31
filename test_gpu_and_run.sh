#!/bin/bash
# ─── Test PyTorch 1.12.1 GPU Support & Run Experiment ───────────────────
# Run this on the server:
#   chmod +x test_gpu_and_run.sh
#   ./test_gpu_and_run.sh
# ──────────────────────────────────────────────────────────────────────────

set -e

cd "$(dirname "$0")"

echo ""
echo "==============================================="
echo "PyTorch 1.12.1 GPU Support Test"
echo "==============================================="
echo ""

# Step 1: Uninstall old PyTorch
echo "[1] Removing old PyTorch installation..."
pip uninstall torch -y --quiet 2>/dev/null || true
echo "    ✓ Done"
echo ""

# Step 2: Install PyTorch 1.12.1 (supports sm_61)
echo "[2] Installing PyTorch 1.12.1 (supports TITAN X Pascal sm_61)..."
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu116 --quiet 2>&1 | grep -v "already satisfied" || true
echo "    ✓ Done"
echo ""

# Step 3: Test GPU
echo "[3] Testing GPU compatibility..."
python << 'PYEOF'
import sys
import torch

print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        with torch.no_grad():
            test_tensor = torch.zeros(1, device='cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"    ✓ GPU WORKS! Device: {device_name}")
        sys.exit(0)
    except Exception as e:
        print(f"    ✗ GPU test failed: {e}")
        sys.exit(1)
else:
    print(f"    ℹ CUDA not available")
    sys.exit(1)
PYEOF

GPU_RESULT=$?
echo ""

# Step 4: Determine device
if [ $GPU_RESULT -eq 0 ]; then
    DEVICE="cuda"
    echo "==============================================="
    echo "GPU is working! Running with CUDA acceleration"
    echo "==============================================="
else
    DEVICE="cpu"
    echo "==============================================="
    echo "GPU not available. Using CPU fallback"
    echo "==============================================="
fi
echo ""

# Step 5: Create tmux session and run experiment
echo "[4] Starting small experiment in tmux session..."
echo "    Device: $DEVICE"
echo ""

mkdir -p results

tmux new-session -d -s exp -x 250 -y 50

tmux send-keys -t exp "cd '$(pwd)' && python run_experiment.py --size small --variant BCAS --device $DEVICE 2>&1 | tee results/test_${DEVICE}_small.txt" Enter

sleep 1

echo "✓ Experiment started in tmux session 'exp'"
echo ""
echo "Monitor progress with:"
echo "  tmux attach-session -t exp"
echo ""
echo "Detach (leave running) with:"
echo "  Ctrl+B then D"
echo ""
echo "Kill session with:"
echo "  tmux kill-session -t exp"
echo ""
echo "Results will be saved to: results/test_${DEVICE}_small.txt"
echo ""

# Auto-attach
echo "Attaching to tmux session..."
sleep 2
tmux attach-session -t exp
