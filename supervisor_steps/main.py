import sys

class Tee:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, filename):
        # Use UTF-8 so ✓, ✗, and non-English characters don't break
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


from torch.cuda.amp import autocast, GradScaler
from funtions import train_with_grpo, evaluate_supervisor, evaluate_supervisor_chained, prepare_pubmedqa_dataset, combined_reward_with_supervisor_penalty, chained_reward

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import os

def main():
    """Main function to orchestrate the entire process."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")
    
    # --- Model and Tokenizer Loading ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}...")
    # Load in default FP32 precision, as mixed precision will be handled by the training loop if enabled.
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded.")
    
    # Move model to device FIRST before evaluation
    model.to(device)
    
    # --- Data Preparation ---
    all_data = prepare_pubmedqa_dataset()
    random.shuffle(all_data)
    eval_data_size = 2
    eval_data = all_data[:eval_data_size]
    train_data = all_data[eval_data_size:]
    print(f"Data prepared. Training examples: {len(train_data)}, Evaluation examples: {len(eval_data)}")


    training_config = {
        'num_iterations': 1,        # Number of times to update the reference model
        'num_steps': 20,           # Batches per iteration. Increase for more training.
        'batch_size': 2,            # Prompts per batch. Decrease if OOM.
        'num_generations': 4,       # Completions per prompt. Decrease if OOM.
        'max_completion_length': 300, # Decrease if OOM.
        'beta': 0.01,               # KL penalty strength
        'learning_rate': 5e-6,      # Optimizer learning rate
        'mu': 2,                    # Number of optimization steps per batch
        'epsilon': 0.2              # PPO clipping value
    }
    
  
    wandb_project = "GRPO-Qwen-PubMedQA-Manual"
    wandb_run_name = "multi-agent-grpo-run"
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=training_config
    )
    print(f"Weights & Biases initialized: project={wandb_project}, run={wandb_run_name}")

    # --- Pre-Training Evaluation ---
    print("\nEvaluating model before fine-tuning...")

    print("\nChained multi-agent evaluation:")
    evaluate_supervisor_chained(model, tokenizer, eval_data, device, max_steps=3)

    # --- Training Configuration ---
    # This config is designed for a single GPU with ~16-24GB VRAM. Adjust if needed.


    # --- Start Training ---
    print("\nStarting GRPO fine-tuning...")
    trained_model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        device=device,
        batch_size=1,            # safer for GPU memory
        num_generations=2,       # safer for GPU memory
        max_completion_length=128,
        reward_function=chained_reward,  # Use chained reward for multi-agent sequences
        use_lora=True            # optional
    )

    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.finish()

    # --- Post-Training Evaluation ---
    print("\nEvaluating model after GRPO fine-tuning...")
    print("Single-agent evaluation:")
    evaluate_supervisor(trained_model, tokenizer, eval_data, device)
    print("\nChained multi-agent evaluation:")
    evaluate_supervisor_chained(trained_model, tokenizer, eval_data, device, max_steps=3)

    # --- Save Final Model ---
    output_dir = "grpo_pubmedqa_finetuned_model"
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")
    wandb.finish()


if __name__ == "__main__":
    main()