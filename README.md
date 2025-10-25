#  GRPO-PubMedQA Manual Training

This project fine-tunes a **Qwen-2.5-0.5B-Instruct** model on the **PubMedQA** dataset using a manually supervised **GRPO (Group Relative Policy Optimization)** pipeline, with subagent using qwen 7b.

---

##  How to Run

###  Option 1 — Using Docker

**Step 1:** Go into the project folder:
```bash
cd supervisor
```

**Step 2:** Build the Docker image:
```bash
docker build -t grpo-pubmedqa:latest .
```

**Step 3:** Run the container with GPU support:
```bash
docker run -it --gpus all \
  -v ${PWD}:/workspace \
  -e WANDB_API_KEY=your_key_here \
  -e WANDB_PROJECT=GRPO-Qwen-PubMedQA-Manual \
  grpo-pubmedqa:latest

```

---

###  Option 2 — Run Locally (Without Docker)

**Step 1:** Install dependencies:
```bash
pip install -r requirements.txt
```

**Step 2:** Set up Weights & Biases (W&B) for experiment tracking:
```bash
export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=GRPO-Qwen-PubMedQA-Manual
```

**Step 3:** Run the training script:
```bash
python main.py
```

---

##  Notes

- Ensure you have a working **GPU + CUDA** setup.  
- **Weights & Biases** is optional but recommended for tracking metrics and losses.  
- You can modify the default model name or dataset path directly inside `main.py` if needed.  
- The model and tokenizer will be automatically downloaded from Hugging Face on first run.

---

##  Example Environment Variables (Windows PowerShell)
```powershell
setx WANDB_API_KEY "your_key_here"
setx WANDB_PROJECT "GRPO-Qwen-PubMedQA-Manual"
```

---

That’s it! 🎯  
You’re ready to train and evaluate your GRPO-based PubMedQA supervisor model.
