#!/bin/bash

# =========================================
# Multi-Agent QA System - Full One-Shot Installer
# =========================================

echo "🚀 Starting full environment setup for Multi-Agent QA System..."
echo "=========================================================="

# --------------------------
# 1️⃣ Update system packages
# --------------------------
echo "📦 Updating system packages..."
apt update && apt upgrade -y
apt install -y build-essential git curl wget unzip zip software-properties-common python3-dev python3-venv

# --------------------------
# 2️⃣ Check Python version
# --------------------------
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 >=3.8"
    exit 1
fi

PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VER="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VER" "$PYTHON_VER" | sort -V | head -n1)" != "$REQUIRED_VER" ]; then
    echo "❌ Python $PYTHON_VER detected. Python >=3.8 is required."
    exit 1
fi

echo "✅ Python $PYTHON_VER detected"

# --------------------------
# 3️⃣ Create virtual environment
# --------------------------
ENV_DIR="$HOME/openthought_env"
echo "🐍 Creating Python virtual environment at $ENV_DIR..."
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Upgrade pip inside venv
pip install --upgrade pip

# --------------------------
# 4️⃣ Detect GPU and install PyTorch
# --------------------------
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ GPU detected, installing CUDA-enabled PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   ⚠️ No GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# --------------------------
# 5️⃣ Install Hugging Face, PEFT, and utilities
# --------------------------
echo "🤗 Installing Hugging Face + LoRA + utilities..."
pip install \
    transformers>=4.30.0 datasets>=2.12.0 tokenizers>=0.13.0 accelerate>=0.20.0 \
    peft>=0.4.0 numpy>=1.21.0 pandas>=2.0.0 requests>=2.31.0 tqdm>=4.65.0 rich>=13.0.0 \
    jupyter>=1.0.0 ipykernel>=6.0.0

# --------------------------
# 6️⃣ Verify installation
# --------------------------
echo "🔍 Verifying installations..."
python3 - <<EOF
import torch, transformers, peft, numpy as np, pandas as pd, requests
print("✅ All core packages imported successfully!")
print(f"   PyTorch version: {torch.__version__}")
print(f"   Transformers version: {transformers.__version__}")
print(f"   PEFT version: {peft.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
EOF

# --------------------------
# 7️⃣ Add virtual environment activation helper
# --------------------------
echo ""
echo "🐍 To activate your Python environment in future sessions, run:"
echo "   source $ENV_DIR/bin/activate"

echo ""
echo "🎉 Full environment setup complete!"
echo "=========================================================="
echo "Next steps:"
echo "1. Pull your repository: git clone <your_repo_url>"
echo "2. Activate the virtual environment: source $ENV_DIR/bin/activate"
echo "3. Run Jupyter notebooks or start training on multi-GPU"
echo ""
echo "Happy coding! 🚀"
