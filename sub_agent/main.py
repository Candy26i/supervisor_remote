import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functions import prepare_pubmedqa_dataset, evaluate_model, train_with_grpo, combined_reward
import random
import os
import wandb


def main():
    """Main function to orchestrate the entire process."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    # --- Model and Tokenizer Loading ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}...")
    # Load in default FP32 precision, as mixed precision will be handled by the training loop if enabled.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded.")
    
    # Move model to device FIRST before evaluation
    model.to(device)

    # --- Data Preparation ---
    all_data = prepare_pubmedqa_dataset()
    random.shuffle(all_data)
    eval_data_size = 20
    eval_data = all_data[:eval_data_size]
    train_data = all_data[eval_data_size:]
    print(f"Data prepared. Training examples: {len(train_data)}, Evaluation examples: {len(eval_data)}")

    # --- Pre-Training Evaluation ---
    print("\nEvaluating model before fine-tuning...")
    before_Acc = evaluate_model(model, tokenizer, eval_data, device)

    # --- Training Configuration ---
    # This config is designed for a single GPU with ~16-24GB VRAM. Adjust if needed.
    training_config = {
        'num_iterations': 1,        # Number of times to update the reference model
        'num_steps': 100,           # Batches per iteration. Increase for more training.
        'batch_size': 2,            # Prompts per batch. Decrease if OOM.
        'num_generations': 4,       # Completions per prompt. Decrease if OOM.
        'max_completion_length': 300, # Decrease if OOM.
        'beta': 0.01,               # KL penalty strength
        'learning_rate': 5e-6,      # Optimizer learning rate
        'mu': 2,                    # Number of optimization steps per batch
        'epsilon': 0.2              # PPO clipping value
    }
    
    # Initialize wandb if API key is set
    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.init(project=os.environ["WANDB_PROJECT"], config=training_config, reinit=True)
    #     print("Weights & Biases initialized.")

    # --- Start Training ---
    print("\nStarting GRPO fine-tuning...")
    trained_model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=combined_reward,
        **training_config
    )
    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.finish()

    # --- Post-Training Evaluation ---
    print("\nEvaluating model after GRPO fine-tuning...")
    after_Acc = evaluate_model(trained_model, tokenizer, eval_data, device)

    # --- Save Final Model ---
    output_dir = "grpo_pubmedqa_finetuned_model"
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")
    print(f"Before Training Accuracy: {before_Acc:.2f}%")
    print(f"After Training Accuracy: {after_Acc:.2f}%")

if __name__ == "__main__":
    main()