#!/bin/bash

# Multi-Agent QA System - Dependency Installation Script
# This script installs all required packages for the GRPO training and multi-agent system

echo "🚀 Installing dependencies for Multi-Agent QA System..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. Python 3.8+ is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU detected - installing PyTorch with CUDA support"
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   No GPU detected - installing CPU-only PyTorch"
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Hugging Face Transformers and related packages
echo "🤗 Installing Hugging Face packages..."
python3 -m pip install transformers>=4.30.0
python3 -m pip install datasets>=2.12.0
python3 -m pip install tokenizers>=0.13.0
python3 -m pip install accelerate>=0.20.0

# Install PEFT for LoRA fine-tuning
echo "🔧 Installing PEFT for LoRA..."
python3 -m pip install peft>=0.4.0

# Install basic data science packages
echo "📊 Installing data science packages..."
python3 -m pip install numpy>=1.21.0
python3 -m pip install pandas>=2.0.0

# Install HTTP requests library
echo "🌐 Installing HTTP libraries..."
python3 -m pip install requests>=2.31.0

# Install optional but useful packages
echo "🛠️  Installing utility packages..."
python3 -m pip install tqdm>=4.65.0
python3 -m pip install rich>=13.0.0

# Install Jupyter for notebook support
echo "📓 Installing Jupyter..."
python3 -m pip install jupyter>=1.0.0
python3 -m pip install ipykernel>=6.0.0

# Install optional packages (commented out by default)
echo "📝 Optional packages (uncomment if needed):"
echo "   # Weights & Biases for experiment tracking:"
echo "   # python3 -m pip install wandb"
echo ""
echo "   # OpenAI API client:"
echo "   # python3 -m pip install openai>=1.0.0"
echo ""
echo "   # LangChain for advanced LLM workflows:"
echo "   # python3 -m pip install langchain>=0.1.0"

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import transformers
import peft
import numpy as np
import pandas as pd
import requests
print('✅ All core packages imported successfully!')
print(f'   PyTorch version: {torch.__version__}')
print(f'   Transformers version: {transformers.__version__}')
print(f'   PEFT version: {peft.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "🎉 Installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Make sure Ollama is installed and running (for local model inference)"
echo "2. Pull a model in Ollama: ollama pull qwen2.5:0.5b-instruct"
echo "3. Run the notebook: jupyter notebook pubmedqa_grpo_supervisor.ipynb"
echo ""
echo "For GPU training, ensure you have:"
echo "- NVIDIA GPU with CUDA 11.8+ support"
echo "- Sufficient VRAM (8GB+ recommended for training)"
echo ""
echo "Happy coding! 🚀"
