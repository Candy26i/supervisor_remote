# ================================================================
# 1️⃣ BASE IMAGE — PyTorch with CUDA 12.1
# ================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ================================================================
# 2️⃣ SYSTEM DEPENDENCIES
# ================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# 3️⃣ PYTHON ENVIRONMENT
# ================================================================
WORKDIR /workspace

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python packages from your requirements.txt
RUN pip install -r requirements.txt

# ================================================================
# 4️⃣ COPY PROJECT FILES
# ================================================================
COPY main.py ./
COPY funtions.py ./
# Optional: copy dataset if it exists in your repo
COPY pubmedqa_5.csv ./

# ================================================================
# 5️⃣ ENVIRONMENT VARIABLES
# ================================================================
# (Adjust as needed)
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
ENV WANDB_PROJECT="GRPO-Qwen-PubMedQA-Manual"

# Hugging Face + torch cache directories
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
RUN mkdir -p $HF_HOME $TORCH_HOME

# ================================================================
# 6️⃣ COMMAND
# ================================================================
# Default command: run your main script
CMD ["python", "main.py"]
