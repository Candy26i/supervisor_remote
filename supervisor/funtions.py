# manual_grpo_pubmedqa.py

# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================

# Basic Python libraries
import random
import copy
import re
import os
import numpy as np
# import wandb # Optional, for logging
import json
import csv
# PyTorch and related libraries
import torch
import torch.nn as nn
torch.cuda.empty_cache()
# Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import os
def set_random_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for consistent results
set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
# Replace with your own key or comment out if not using wandb
# os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
# os.environ["WANDB_PROJECT"] = "GRPO-Qwen-PubMedQA-Manual"

# ==============================================================================
# SECTION 2: PROMPT AND DATA PREPARATION (ADAPTED FOR PUBMEDQA)
# ==============================================================================
# ---- SUPERVISOR SIDE ----
SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor routing requests to specialized agents.

Reply STRICTLY in the following format:
Agent: <name>
<one short sentence explaining why>

The agent name must be exactly one of:
question_understanding, context_analysis, reasoning, answering

IMPORTANT: Keep your response SHORT. Only output the agent name and one brief explanation sentence. Do not add any additional reasoning, answers, or explanations beyond this format."""

def build_supervisor_prompt(example: dict) -> str:
    """Build the supervisor routing prompt from a dataset row."""
    return (
        SUPERVISOR_SYSTEM_PROMPT
        + "\n\n"
        + f"Given the problem:\n{example['problem']}\n\n"
        + f"Context:\n{example['context']}\n\n"
        + "Please choose ONE next agent to call "
          "from: question_understanding, context_analysis, reasoning, answering.\n\n"
        + "Reply STRICTLY in the form:\nAgent: <name>\nThen explain why.\n\n"
        + "REMEMBER: Only output the agent name and one brief explanation sentence. Do not add reasoning sections, answers, or any other content."
    )



VALID_AGENTS = {
    "question_understanding",
    "context_analysis",
    "reasoning",
    "answering",
}

# Strict: must appear as a line that begins with "Agent:" then a valid name
_AGENT_LINE_RE = re.compile(
    r"(?im)^\s*Agent:\s*(question_understanding|context_analysis|reasoning|answering)\b"
)

def parse_supervisor_choice(supervisor_msg: str, fallback_names=None):
    """
    Returns chosen agent name (normalized lowercase) or None.
    1) Strictly parse the 'Agent: <name>' line.
    2) Fallback: keyword presence from allowed list in message body.
    """
    m = _AGENT_LINE_RE.search(supervisor_msg)
    if m:
        return m.group(1).lower()

    # Fallback: scan allowed names if the strict line is missing
    names = fallback_names or VALID_AGENTS
    msg_low = supervisor_msg.lower()
    for name in names:
        # match whole token (avoid partial overlaps)
        if re.search(rf"\b{name}\b", msg_low):
            return name
    return None

# Define the structured prompt format for our task
SYSTEM_PROMPT = """You are an expert biomedical researcher. Your task is to answer a question based on a provided context.
First, write out a step-by-step reasoning process within <reasoning> tags.
Then, provide the final answer (either "yes" or "no") within <answer> tags.
"""

def build_prompt(messages):
    """Builds a single prompt string from a list of messages."""
    return "\n".join([msg["content"].strip() for msg in messages])

def prepare_pubmedqa_dataset(csv_file_path="pubmedqa_5.csv"):
    """Loads and prepares the PubMedQA dataset from a CSV file."""
    formatted_data = []
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem = row["question"].strip()
            context_info = row["context"].strip()
            answer = row["final_decision"].strip().lower()

            user_content = f"Context:\n{context_info}\n\nQuestion:\n{problem}"
            prompt_str = build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ])

            formatted_data.append({
                "problem": problem,
                "context": context_info,
                "prompt": prompt_str,
                "answer": answer
            })
    return formatted_data

# ==============================================================================
# SECTION 2.5: Ollama calling the subagent specified by the supervisor
# ==============================================================================
import requests
import json as _json

OLLAMA_MODEL = "qwen2.5:0.5b-instruct"   # or your tag e.g. "qwen2.5-0.5b-instruct"

AGENT_PROMPTS = {
    "question_understanding": """You are the Question-Understanding agent.
Given the following problem and context, clarify the exact question to be answered and restate it precisely.
Based on your analysis, provide your best answer to the question.

Format your response as:
Answer: yes/no/maybe

IMPORTANT: Always end with "Answer: yes/no/maybe" based on your understanding of the question.""",

    "context_analysis": """You are the Context-Analysis agent.
Analyze the provided context: extract key facts, contradictions, and relevance to the question.
Based on your analysis of the context, provide your best answer to the question.

Format your response as:
<analysis>Analysis of the context and key facts</analysis>
Answer: yes/no/maybe

IMPORTANT: Always end with "Answer: yes/no/maybe" based on your analysis of the context.""",

    "reasoning": """You are the Reasoning agent.
Do step-by-step reasoning combining the question and the context to reach a conclusion.
Based on your reasoning, provide your best answer to the question.

Format your response as:
<reasoning>Step-by-step reasoning combining question and context</reasoning>
Answer: yes/no/maybe

IMPORTANT: Always end with "Answer: yes/no/maybe" based on your reasoning.""",

    "answering": """You are the Answering agent.
Given the problem and the context, provide your final answer.
This is the final decision-making agent.

Format your response as:
Answer: yes/no/maybe

IMPORTANT: Always end with "Answer: yes/no/maybe" as your final answer to the question."""
}

def build_subagent_prompt(agent_name: str, example: dict) -> str:
    """Build the prompt for a specific subagent from a dataset row."""
    sys = AGENT_PROMPTS[agent_name]
    return (
        f"{sys}\n\n"
        f"Problem:\n{example['problem']}\n\n"
        f"Context:\n{example['context']}\n\n"
    )

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

# Load once globally (outside training loop)

HF_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model_hf = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

def call_ollama(model_name, messages, temperature=0.8, top_p=0.95, max_tokens=256):
    """
    Calls the local model and returns only the newly generated text
    (excluding the input prompt).
    """
    # Combine messages into one text block
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model_hf.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    # Decode only the NEW tokens (excluding the input prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decoded.strip()


def run_subagent(agent_name: str, example: dict):
    """Execute the chosen agent with Ollama and extract final yes/no if present."""
    print(f"\n[DEBUG] Running sub-agent: {agent_name}")
    print(f"[DEBUG] Example keys: {list(example.keys())}")
    
    prompt = build_subagent_prompt(agent_name, example)
    print(f"[DEBUG] Generated prompt length: {len(prompt)}")
    
    # Convert string prompt to message list format for call_ollama
    messages = [{"role": "user", "content": prompt}]
    print(f"[DEBUG] Calling Ollama with model: {OLLAMA_MODEL}")
    
    try:
        text = call_ollama(OLLAMA_MODEL, messages)
        print(f"[DEBUG] Ollama response length: {len(text)}")
        print(f"[DEBUG] Ollama response: {text[:200]}...")  # First 200 chars
        
        # Reuse your stricter extraction:
        ans = extract_answer_from_model_output(text)  # returns "yes"/"no"/None
        print(f"[DEBUG] Extracted answer: {ans}")
        return text, ans
    except Exception as e:
        print(f"[ERROR] Failed to call Ollama: {e}")
        return "", None



# ==============================================================================
# SECTION 3: REWARD FUNCTIONS (ADAPTED FOR PUBMEDQA)
# ==============================================================================

def extract_answer_from_model_output(text):
    """Extracts the value from the 'Answer: yes/no/maybe' format in the text."""
    import re
    
    # First try to find "Answer: yes/no/maybe" pattern
    answer_pattern = r'Answer:\s*(yes|no|maybe)\b'
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    
    # Fallback: try to find answer tags (for backward compatibility)
    answer_tag_pattern = r'<answer>\s*(yes|no|maybe)\s*</answer>'
    matches = re.findall(answer_tag_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    
    # Final fallback: look for yes/no/maybe in the text
    text_lower = text.lower()
    if "maybe" in text_lower:
        return "maybe"
    elif "yes" in text_lower and "no" not in text_lower:
        return "yes"
    elif "no" in text_lower and "yes" not in text_lower:
        return "no"
    elif "yes" in text_lower and "no" in text_lower:
        # If both are present, look for the last occurrence
        yes_pos = text_lower.rfind("yes")
        no_pos = text_lower.rfind("no")
        return "yes" if yes_pos > no_pos else "no"
    
    return None

def pubmedqa_correctness_reward(completions, answer, **kwargs):
    """Assigns a reward based on the correctness of the 'yes'/'no' answer."""
    responses = [comp[0]['content'] for comp in completions]
    extracted_answers = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for extracted, expected in zip(extracted_answers, answer):
        if extracted and extracted == expected:
            rewards.append(2.0)  # High reward for an exact match
        else:
            rewards.append(0.0)  # No reward for wrong or missing answer
    return rewards

def format_reward(completions, **kwargs):
    """Assigns a reward for adhering to the desired XML format."""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score) # Max format score = 0.8
    return rewards

def combined_reward(prompts, completions, answer, **kwargs):
    """Combines correctness and format rewards."""
    correctness_scores = pubmedqa_correctness_reward(completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    combined_rewards = [c_score + f_score for c_score, f_score in zip(correctness_scores, format_scores)]
    return combined_rewards
# ==============================================================================
# SECTION 3.5: CORE GRPO/PPO LOGIC Supervisor
# ==============================================================================


def selective_log_softmax(logits, input_ids):
    """Computes log probabilities for specific tokens."""
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """Computes the log probabilities for a batch of tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """Creates a mask for completion tokens, stopping after the first EOS token."""
    is_eos = completion_ids == eos_token_id
    # Find the index of the first EOS token for each sequence
    eos_indices = torch.argmax(is_eos.int(), dim=1)
    # If no EOS is found, argmax returns 0. We need to handle this.
    # We set the index to max_length if no EOS is found.
    eos_indices[~is_eos.any(dim=1)] = completion_ids.size(1)

    # Create a range tensor to compare with indices
    seq_indices = torch.arange(completion_ids.size(1), device=completion_ids.device).expand_as(completion_ids)
    
    # The mask is 1 for all tokens up to and including the first EOS
    mask = (seq_indices <= eos_indices.unsqueeze(1)).int()
    return mask

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=128):
    """Generates multiple completions for each prompt."""
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    # set once after loading the tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    
    prompt_length = prompt_ids.size(1)
    
    # Repeat prompts to generate multiple completions in one batch
    repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    outputs = model.generate(
        repeated_prompt_ids,
        attention_mask=repeated_prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """Generates data for GRPO rollouts including completions, log probabilities, and debug traces."""
    
    print(f"\n[DEBUG] === Generating GRPO rollouts for {len(batch_samples)} samples ===")

    # prompts = [sample["prompt"] for sample in batch_samples]
    prompts = [build_supervisor_prompt(sample) for sample in batch_samples]

    print(f"[DEBUG] Prompts in generate_rollout_data: {prompts}")
    answers = [sample["answer"] for sample in batch_samples]

    # ---- 1️⃣ Generate completions ----
    print(f"[DEBUG] Generating {num_generations} completions per sample...")
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )

        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
        completion_attn = (completion_ids != tokenizer.pad_token_id).long()

        input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([repeated_prompt_mask, completion_attn], dim=1)
        logits_to_keep = completion_ids.size(1)

        policy_model = model.module if isinstance(model, nn.DataParallel) else model
        reference_model = ref_model.module if isinstance(ref_model, nn.DataParallel) else ref_model

        old_log_probs = compute_log_probs(policy_model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(reference_model, input_ids, attention_mask, logits_to_keep)

    # ---- 2️⃣ Decode supervisor outputs ----
    texts = tokenizer.batch_decode(completion_ids.detach().cpu(), skip_special_tokens=True)
    print(f"[DEBUG] Decoded {len(texts)} supervisor outputs.")

    chosen_agents, sub_texts, action_masks = [], [], []

    # ---- 3️⃣ Process each supervisor output ----
    for i, text in enumerate(texts):
        base_idx = i // num_generations
        sample = batch_samples[base_idx]

        # Parse supervisor decision
        chosen = parse_supervisor_choice(text)
        chosen_agents.append(chosen if chosen in VALID_AGENTS else None)

        print("\n----------------------------------------")
        print(f"[DEBUG] Sample {base_idx + 1}/{len(batch_samples)}, Generation {i + 1}")
        print(f"[DEBUG] Supervisor Output Preview: {text[:250].replace(chr(10), ' ')}{'...' if len(text) > 250 else ''}")
        print(f"[DEBUG] Parsed Agent Choice: {chosen}")

        # ---- 4️⃣ Run corresponding sub-agent ----
        if chosen in VALID_AGENTS:
            print(f"[DEBUG] Calling Sub-Agent: {chosen}")
            sub_out_text, _ = run_subagent(chosen, sample)
            sub_texts.append(sub_out_text)
            print(f"[DEBUG] Sub-Agent '{chosen}' completed.")
        else:
            print("[WARN] Invalid or missing agent choice — skipping sub-agent.")
            sub_texts.append("")

        # ---- 5️⃣ Build action mask ----
        comp_row = completion_ids[i]
        valid_len = int((comp_row != tokenizer.pad_token_id).sum().item())
        L = min(8, valid_len)
        mask = torch.zeros_like(comp_row, dtype=torch.long)
        if L > 0:
            mask[:L] = 1
        action_masks.append(mask)

    action_mask = torch.stack(action_masks, dim=0).to(input_ids.device)

    # ---- 6️⃣ Prepare rollout dictionary ----
    formatted_completions = [[{'content': t}] for t in sub_texts]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    print(f"\n[DEBUG] === Rollout Generation Complete ===")
    print(f"[DEBUG] Total Generated: {len(formatted_completions)} completions")
    print(f"[DEBUG] Valid agent decisions: {sum(a is not None for a in chosen_agents)} / {len(chosen_agents)}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "action_mask": action_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def grpo_loss(model, ref_model, rollout_data, reward_function, beta=0.01, epsilon=0.2, target_kl=0.01):
    """Computes the GRPO loss for updating the policy model with dynamic KL controller."""
    device = next(model.parameters()).device
    
    # Unpack rollout data
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    action_mask = rollout_data["action_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    
    # Compute current log probs
    policy_model = model.module if isinstance(model, nn.DataParallel) else model
    token_log_probs = compute_log_probs(policy_model, input_ids, attention_mask, logits_to_keep)
    
    # Calculate ratio and rewards
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"], 
            completions=rollout_data["formatted_completions"], # subagent outputs
            answer=rollout_data["repeated_answers"]
        ),
        dtype=torch.float32,
        device=device
    )

    # Standardize rewards at the group level (GRPO)
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    
    # Group rewards by prompts (num_prompts, k)
    num_prompts = batch_size
    k = num_generations
    group_rewards = rewards.view(num_prompts, k)
    mean_R = group_rewards.mean(dim=-1, keepdim=True)
    std_R = group_rewards.std(dim=-1, keepdim=True)
    advantages = (group_rewards - mean_R) / (std_R + 1e-8)
    
    # Additional advantages normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.view(-1).unsqueeze(1) # Flatten back for token-wise multiplication
    
    # PPO Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    
    # Dynamic KL controller (same as PPO2)
    with torch.no_grad():
        # Calculate KL divergence between old and new policies
        kl = (token_log_probs - old_log_probs).mean()
        
        # Adjust beta based on KL divergence relative to target
        beta = beta * (kl / target_kl).clamp(0.5, 2.0)
    
    # KL Penalty with dynamic beta
    kl_div = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    
    # Combine and mask the loss
    per_token_loss = surrogate_loss - beta * kl_div
    # We only care about the loss for the completion tokens
    masked_loss = per_token_loss * action_mask.to(per_token_loss.dtype)
    loss = -(masked_loss.sum() / action_mask.sum().clamp_min(1))
    
    avg_reward = rewards.mean().item()
    return loss, avg_reward

# ==============================================================================
# SECTION 5: TRAINING LOOP (IMITATED AND ADAPTED FOR SINGLE/MULTI GPU)
# ==============================================================================
import torch
from torch.cuda.amp import autocast, GradScaler
import copy
import random

def train_with_grpo(
    model,
    tokenizer,
    train_data,
    num_iterations=1,
    num_steps=100,
    batch_size=1,                 # reduce batch_size for GPU memory
    num_generations=2,            # reduce generations
    max_completion_length=128,    # reduce completion length
    beta=0.1,
    learning_rate=5e-6,
    mu=3,
    epsilon=0.2,
    target_kl=0.01,               # target KL divergence for dynamic controller
    reward_function=pubmedqa_correctness_reward,
    device=None,
    use_lora=True                # optional flag to enable LoRA
):
    """Memory-safe GRPO training loop with mixed precision and optional LoRA."""

    # 1️⃣ Device setup
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # 2️⃣ Optional LoRA
    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj","k_proj","v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied to q/k/v projections.")

    model.to(device)

    # 3️⃣ Mixed precision scaler
    scaler = GradScaler()

    # Outer loop for updating the reference model
    for iteration in range(num_iterations):
        print(f"\n--- Starting GRPO Iteration {iteration + 1}/{num_iterations} ---")

        # Reference model
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.to(device)
        print("Reference model created.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Inner loop for batch updates
        for step in range(num_steps):
            n = min(batch_size, len(train_data))
            batch_samples = random.sample(train_data, n)

            # 1️⃣ Generate rollouts
            rollout_data = generate_rollout_data(
                model,
                ref_model,
                tokenizer,
                batch_samples,
                num_generations,
                max_completion_length
            )

            # 2️⃣ PPO-style updates with mixed precision
            for _ in range(mu):
                optimizer.zero_grad()
                
                # Mixed precision context
                with autocast():  
                    loss, avg_reward = grpo_loss(
                        model,
                        ref_model,
                        rollout_data,
                        reward_function,
                        beta=beta,
                        epsilon=epsilon,
                        target_kl=target_kl
                    )

                # Scaled backward pass
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_value = loss.item()
                # Clear memory after each inner step
                del loss
                torch.cuda.empty_cache()

            # Clear rollout_data after all inner iterations are done
            del rollout_data
            torch.cuda.empty_cache()

            print(f"Iter {iteration+1}, Step {step+1}/{num_steps}, Avg Reward: {avg_reward:.2f}")
            log_data = {
                "iteration": iteration + 1,
                "step": step + 1,
                "avg_reward": avg_reward,
                "loss": loss_value,
            }


            wandb.log(log_data)
        
        # Update reference model with exponential moving average
        print("Updating reference model with exponential moving average...")
        with torch.no_grad():
            for ref_param, actor_param in zip(ref_model.parameters(), model.parameters()):
                ref_param.data = 0.95 * ref_param.data + 0.05 * actor_param.data
        
    return model


# ==============================================================================
# SECTION 6: EVALUATION (ADAPTED FOR PUBMEDQA)
# ==============================================================================
def evaluate_supervisor(model, tokenizer, eval_examples, device=None, max_supervisor_new_tokens=64):
    """Evaluates accuracy by: supervisor routes -> sub-agent answers -> compare to gold yes/no."""
    if device is None:
        device = next(model.parameters()).device

    print(f"[DEBUG] Evaluation using model type: {type(model)}")
    print(f"[DEBUG] Model device: {device}")
    print(f"[DEBUG] Model training mode before eval(): {model.training}")
    
    # Check if this is a LoRA model
    if hasattr(model, 'peft_config'):
        print(f"[DEBUG] LoRA model detected with config: {model.peft_config}")
    else:
        print(f"[DEBUG] Standard model (no LoRA)")
    
    model.eval()
    correct, total = 0, len(eval_examples)
    print("\n" + "="*50)
    print(f"STARTING SUPERVISOR EVALUATION ON {total} EXAMPLES")
    print("="*50)

    for ex in eval_examples:
        # 1) Build supervisor prompt from the original user prompt
        sup_prompt = build_supervisor_prompt(ex)
        expected = ex["answer"]

        # 2) Supervisor generates a routing decision
        inputs = tokenizer.encode(sup_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_supervisor_new_tokens,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # sup_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 3) Parse agent choice (default to 'answering' if bad format)
        new_tokens = outputs[0][inputs.shape[1]:]
        sup_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        agent = parse_supervisor_choice(sup_response)
        print(f"\n[DEBUG] Parsed agent choice: {agent}")
        print(f"[DEBUG] Valid agents: {VALID_AGENTS}")
        print(f"[DEBUG] Agent in valid agents: {agent in VALID_AGENTS}")

        # 4) Run sub-agent via Ollama on the ORIGINAL user prompt
        print(f"\n[DEBUG] About to call sub-agent: {agent}")
        if agent in VALID_AGENTS:
            print(f"[DEBUG] Agent is valid, calling run_subagent...")
            sub_text, pred = run_subagent(agent, ex)
            print(f"[DEBUG] Sub-agent call completed. Pred: {pred}")
        else:
            print(f"[WARNING] Invalid agent '{agent}', skipping sub-agent call")
            sub_text, pred = "", None

        # 5) Score using your unchanged extractor
        # pred = extract_answer_from_model_output(sub_text)
        is_correct = (pred == expected)
        correct += int(is_correct)

        # Optional logging
        print("\n--- Example ---")
        print(f"Expected: {expected} | Pred: {pred} | Agent: {agent} | Correct: {'✓' if is_correct else '✗'}")
        print(f"[Supervisor Message]\n{sup_response}")
        print(f"[Sub-agent Response]\n{sub_text}")
        print("-"*50)

    acc = 100.0 * correct / max(total, 1)
    print(f"\nEvaluation Complete. Accuracy: {acc:.2f}% ({correct}/{total})")
    wandb.log({"evaluation_accuracy": acc})

    print("="*50)
    model.train()
    return acc


# ==============================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# ==============================================================================
