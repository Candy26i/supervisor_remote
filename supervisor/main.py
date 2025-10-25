import sys
import os
import random
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from funtions import train_with_grpo, evaluate_supervisor, prepare_pubmedqa_dataset


# ==============================================================
# Tee class for logging both to terminal and file
# ==============================================================
class Tee:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, filename: str):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.file = open(filename, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.__stdout__

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return self.stdout.isatty()

    def fileno(self):
        return self.stdout.fileno()


# ==============================================================
# Main function
# ==============================================================
def main():
    """Main function to orchestrate GRPO fine-tuning on PubMedQA."""
    # ---------- Logging ----------
    log_path = "logs/grpo_pubmedqa_training.txt"
    sys.stdout = sys.stderr = Tee(log_path)
    print(f"[INFO] Logging to: {log_path}")

    # ---------- Device setup ----------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    

    # ---------- Model & Tokenizer ----------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[INFO] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    print("[INFO] Model and tokenizer loaded successfully.")

    # ---------- Dataset ----------
    try:
        all_data = prepare_pubmedqa_dataset()
        random.shuffle(all_data)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    eval_data_size = 1
    eval_data = all_data[:eval_data_size]
    train_data = all_data[eval_data_size:]
    print(f"[INFO] Dataset prepared: {len(train_data)} train, {len(eval_data)} eval")

    # ---------- Training Config ----------
    training_config = {
        "num_iterations": 1,
        "num_steps": 100,
        "batch_size": 1,               # safer for 24GB GPUs
        "num_generations": 2,
        "max_completion_length": 128,
        "beta": 0.01,
        "learning_rate": 5e-6,
        "mu": 2,
        "epsilon": 0.2,
    }

    # ---------- Initialize Weights & Biases ----------
    wandb_project = "GRPO-Qwen-PubMedQA-Manual"
    wandb_run_name = "multi-agent-grpo-run"
    try:
        wandb.init(project=wandb_project, name=wandb_run_name, config=training_config)
        print(f"[INFO] W&B initialized: {wandb_project}/{wandb_run_name}")
    except Exception as e:
        print(f"[WARN] W&B init failed ({e}). Continuing without logging...")

    # ---------- Pre-training Evaluation ----------
    print("\n[STAGE] Evaluating model before fine-tuning...")
    evaluate_supervisor(model, tokenizer, eval_data, device)

    # ---------- Training ----------
    print("\n[STAGE] Starting GRPO fine-tuning...")
    try:
        trained_model = train_with_grpo(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            device=device,
            batch_size=training_config["batch_size"],
            num_generations=training_config["num_generations"],
            max_completion_length=training_config["max_completion_length"],
            use_lora=True,
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    torch.cuda.empty_cache()

    # ---------- Post-training Evaluation ----------
    print("\n[STAGE] Evaluating model after GRPO fine-tuning...")
    trained_model.eval().to(device)
    evaluate_supervisor(trained_model, tokenizer, eval_data, device)

    # ---------- Save Model ----------
    output_dir = "grpo_pubmedqa_finetuned_model"
    print(f"\n[STAGE] Saving fine-tuned model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Model saved successfully.")

    if wandb.run:
        wandb.finish()

    print("\n[INFO] Training pipeline completed successfully.")


# ==============================================================
# Entry Point
# ==============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}")
        raise
