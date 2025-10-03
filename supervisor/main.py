from torch.cuda.amp import autocast, GradScaler
from funtions import train_with_grpo, evaluate_supervisor, prepare_pubmedqa_dataset

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """Main function to orchestrate the entire process."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    # --- Model and Tokenizer Loading ---
    model_name = "Qwen/Qwen2.5-7B-Instruct"
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
    eval_data_size = 400
    eval_data = all_data[:eval_data_size]
    train_data = all_data[eval_data_size:]
    print(f"Data prepared. Training examples: {len(train_data)}, Evaluation examples: {len(eval_data)}")

    # --- Pre-Training Evaluation ---
    print("\nEvaluating model before fine-tuning...")
    acc_before =evaluate_supervisor(model, tokenizer, eval_data, device)

 
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
        device=device,
        batch_size=1,            # safer for GPU memory
        num_generations=2,       # safer for GPU memory
        max_completion_length=128,
        use_lora=True            # optional
    )

    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.finish()

    # --- Post-Training Evaluation ---
    print("\nEvaluating model after GRPO fine-tuning...")
    acc_after = evaluate_supervisor(trained_model, tokenizer, eval_data, device)

    # --- Save Final Model ---
    output_dir = "grpo_pubmedqa_finetuned_model"
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")
    print(f"Accuracy before training: {acc_before:.2f}%")
    print(f"Accuracy after training: {acc_after:.2f}%")

if __name__ == "__main__":
    main()