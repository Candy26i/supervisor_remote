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
Clarify the exact question to be answered and restate it precisely.
Provide your analysis of what the question is asking.

Format your response as:
<understanding>Your analysis of what the question is asking</understanding>""",

    "context_analysis": """You are the Context-Analysis agent.
Analyze the context: extract key facts, contradictions, and relevance to the question.
Provide your analysis of the context.

Format your response as:
<analysis>Analysis of the context and key facts</analysis>""",

    "reasoning": """You are the Reasoning agent.
Do step-by-step reasoning combining the question and the context to reach a conclusion.
Provide your reasoning process.

Format your response as:
<reasoning>Step-by-step reasoning</reasoning>""",

    "answering": """You are the Answering agent.
Given the problem and the context, and the history of the conversation, provide your final answer.
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
        "Follow the specified XML output strictly."
    )

def build_subagent_prompt_with_history(agent_name: str, example: dict, conversation_history: list = None) -> str:
    """Build the prompt for a specific subagent with conversation history."""
    sys = AGENT_PROMPTS[agent_name]
    
    # Start with the agent system prompt
    prompt = f"{sys}\n\n"
    
    # Add conversation history if available
    if conversation_history:
        prompt += "Based on the following conversation history:\n"
        for i, (agent, output) in enumerate(conversation_history, 1):
            prompt += f"{i}. {agent}: {output}\n"
        prompt += "\n"
    else:
        # Only include problem and context if no history (first agent call)
        prompt += f"Problem:\n{example['problem']}\n\n"
        prompt += f"Context:\n{example['context']}\n\n"
    
    prompt += "Follow the specified XML output strictly."
    return prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

# Load once globally (outside training loop)
HF_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

def call_ollama(model_name, messages, temperature=0.8, top_p=0.95, max_tokens=256):
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    # Extract only the newly generated tokens (not the input prompt)
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded


def run_subagent(agent_name: str, example: dict, conversation_history: list = None):
    """Execute the chosen agent with conversation history and extract final yes/no if present."""
    print(f"\n[DEBUG] Running sub-agent: {agent_name}")
    print(f"[DEBUG] Example keys: {list(example.keys())}")
    
    # Build prompt with conversation history
    prompt = build_subagent_prompt_with_history(agent_name, example, conversation_history)
    print(f"[DEBUG] Generated prompt length: {len(prompt)}")
    
    # Convert string prompt to message list format for call_ollama
    messages = [{"role": "user", "content": prompt}]
    print(f"[DEBUG] Calling Ollama with model: {OLLAMA_MODEL}")
    
    try:
        text = call_ollama(OLLAMA_MODEL, messages)
        print(f"[DEBUG] Ollama response length: {len(text)}")
        print(f"[DEBUG] Ollama response: {text}")  # Clean response without input prompt
        
        # Only extract answer from the answering agent
        if agent_name == "answering":
            ans = extract_answer_from_model_output(text)  # returns "yes"/"no"/None
            print(f"[DEBUG] Extracted answer: {ans}")
        else:
            ans = None  # Non-answering agents don't provide answers
            print(f"[DEBUG] Non-answering agent, no answer extraction")
        
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

def supervisor_token_penalty_reward(completions, answer, supervisor_responses=None, max_tokens=50, penalty_rate=0.01, **kwargs):
    """Adds token count penalty for supervisor responses to encourage conciseness."""
    if supervisor_responses is None:
        return [0.0] * len(completions)
    
    rewards = []
    for response in supervisor_responses:
        # Count tokens in supervisor response
        token_count = len(response.split())  # Simple word count as token approximation
        
        # Calculate penalty: penalty_rate * (tokens - max_tokens) if over limit
        if token_count > max_tokens:
            penalty = penalty_rate * (token_count - max_tokens)
        else:
            penalty = 0.0
        
        rewards.append(-penalty)  # Negative reward (penalty)
    
    return rewards

def combined_reward_with_supervisor_penalty(prompts, completions, answer, supervisor_responses=None, **kwargs):
    """Combines correctness, format, and supervisor token penalty rewards."""
    # Get base rewards
    correctness_scores = pubmedqa_correctness_reward(completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    
    # Get supervisor token penalty
    supervisor_penalties = supervisor_token_penalty_reward(
        completions=completions, 
        answer=answer, 
        supervisor_responses=supervisor_responses
    )
    
    # Combine all rewards
    combined_rewards = [
        c_score + f_score + s_penalty 
        for c_score, f_score, s_penalty in zip(correctness_scores, format_scores, supervisor_penalties)
    ]
    return combined_rewards

def chained_reward(final_answers, expected_answers, supervisor_responses=None, max_tokens=50, penalty_rate=0.01, **kwargs):
    """Reward function for chained multi-agent sequences."""
    rewards = []
    
    for final_answer, expected_answer in zip(final_answers, expected_answers):
        # Base reward for correctness
        if final_answer and final_answer == expected_answer:
            base_reward = 2.0
        else:
            base_reward = 0.0
        
        rewards.append(base_reward)
    
    # Add supervisor token penalty if responses provided
    if supervisor_responses:
        supervisor_penalties = supervisor_token_penalty_reward(
            completions=[],  # Not used for chained reward
            answer=expected_answers,
            supervisor_responses=supervisor_responses,
            max_tokens=max_tokens,
            penalty_rate=penalty_rate
        )
        
        # Combine rewards with penalties
        combined_rewards = [
            base_reward + penalty 
            for base_reward, penalty in zip(rewards, supervisor_penalties)
        ]
        return combined_rewards
    
    return rewards
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
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
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
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """Generates data for GRPO rollouts including completions and log probabilities."""
    prompts = [sample["prompt"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]
    
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        
        # We need the original prompts repeated for log prob calculation
        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
        
        completion_attn = (completion_ids != tokenizer.pad_token_id).long()
        input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([repeated_prompt_mask, completion_attn], dim=1)
        logits_to_keep = completion_ids.size(1)
        # compute_log_probs needs a model on a single device, so we use .module
        # if it is wrapped in DataParallel
        policy_model = model.module if isinstance(model, nn.DataParallel) else model
        reference_model = ref_model.module if isinstance(ref_model, nn.DataParallel) else ref_model

        old_log_probs = compute_log_probs(policy_model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(reference_model, input_ids, attention_mask, logits_to_keep)
    texts = tokenizer.batch_decode(completion_ids.detach().cpu(), skip_special_tokens=True)

    # chosen agents + subagent outputs (to feed your reward fn)
    chosen_agents = []
    sub_texts = []

    # simple action mask: first few non-pad tokens (focus gradients on routing decision)
    action_masks = []
    for i, text in enumerate(texts):
        chosen = parse_supervisor_choice(text)
        chosen_agents.append(chosen if chosen in VALID_AGENTS else None)

        # Map to base sample (which problem/context)
        base_idx = i // num_generations
        sample_i = batch_samples[base_idx]   # get the full sample dict

        if chosen in VALID_AGENTS:
            sub_out_text, _ = run_subagent(chosen, sample_i, conversation_history=None)  # pass the full sample dict
            sub_texts.append(sub_out_text)
        else:
            sub_texts.append("")

        # Build a small front-span mask over completion tokens
        comp_ids_row = completion_ids[i]
        valid_len = int((comp_ids_row != tokenizer.pad_token_id).sum().item())
        L = min(8, valid_len)   # first 8 tokens
        m = torch.zeros_like(comp_ids_row, dtype=torch.long)
        if L > 0:
            m[:L] = 1
        action_masks.append(m)

    action_mask = torch.stack(action_masks, dim=0).to(input_ids.device)

    # Your reward function expects this shape:
    formatted_subagent_completions = [[{'content': t}] for t in sub_texts]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,             # kept (unused for loss now)
        "action_mask": action_mask,                     # <-- NEW: use this in loss
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_subagent_completions,  # <-- feed to your reward
        "supervisor_responses": texts,                  # <-- NEW: supervisor responses for token penalty
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def generate_chained_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length, max_steps=3):
    """Generates chained multi-agent rollout data for GRPO training."""
    prompts = [sample["prompt"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]
    
    all_input_ids = []
    all_attention_masks = []
    all_old_log_probs = []
    all_ref_log_probs = []
    all_action_masks = []
    all_supervisor_responses = []
    all_final_answers = []
    all_repeated_prompts = []
    all_repeated_answers = []
    
    with torch.no_grad():
        for sample_idx, sample in enumerate(batch_samples):
            for gen_idx in range(num_generations):
                print(f"[DEBUG] Generating chained rollout for sample {sample_idx}, generation {gen_idx}")
                
                # Simulate chained conversation
                conversation_history = []
                final_answer = None
                supervisor_responses = []
                
                for step in range(max_steps):
                    # 1) Build supervisor prompt with conversation history
                    sup_prompt = build_supervisor_prompt(sample, conversation_history)
                    print(f"[DEBUG] Supervisor prompt (rollout):\n{sup_prompt}")
                    print(f"[DEBUG] Prompt length: {len(sup_prompt)} characters")
                    
                    # 2) Generate supervisor response (no gradients for rollout generation)
                    inputs = tokenizer(sup_prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=64,
                            temperature=0.1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    
                    # Extract supervisor response
                    input_length = inputs.shape[1]
                    new_tokens = outputs[0][input_length:]
                    sup_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Clean up supervisor response
                    if "Agent:" in sup_response:
                        sup_response = sup_response[sup_response.find("Agent:"):]
                    
                    supervisor_responses.append(sup_response)
                    
                    # Parse agent choice
                    agent = parse_supervisor_choice(sup_response)
                    
                    # Force answering agent if max_steps reached
                    if step == max_steps - 1:
                        agent = "answering"
                    
                    # Run sub-agent
                    if agent in VALID_AGENTS:
                        sub_text, pred = run_subagent(agent, sample, conversation_history)
                        
                        # Add to conversation history
                        clean_decision = re.sub(r"(?im)^.*Agent:\s*", "", sup_response).strip()
                        conversation_history.append(("supervisor", f"Chose {agent}: {clean_decision}"))
                        conversation_history.append((agent, sub_text))
                        
                        # If answering agent, we're done
                        if agent == "answering":
                            final_answer = pred
                            break
                    else:
                        # Invalid agent, force answering
                        agent = "answering"
                        sub_text, pred = run_subagent(agent, sample, conversation_history)
                        clean_decision = re.sub(r"(?im)^.*Agent:\s*", "", sup_response).strip()
                        conversation_history.append(("supervisor", f"Chose {agent}: {clean_decision}"))
                        conversation_history.append((agent, sub_text))
                        final_answer = pred
                        break
                
                # Store the chained sequence data
                all_supervisor_responses.extend(supervisor_responses)
                
                # Store all supervisor responses for training
                for sup_response in supervisor_responses:
                    # Tokenize supervisor response
                    inputs = tokenizer(sup_response, return_tensors="pt").to(model.device)
                    all_input_ids.append(inputs.input_ids)
                    all_attention_masks.append(inputs.attention_mask)
                    
                    # Compute log probs for this response
                    policy_model = model.module if isinstance(model, nn.DataParallel) else model
                    reference_model = ref_model.module if isinstance(ref_model, nn.DataParallel) else ref_model
                    
                    old_log_probs = compute_log_probs(policy_model, inputs.input_ids, inputs.attention_mask, inputs.input_ids.shape[1])
                    ref_log_probs = compute_log_probs(reference_model, inputs.input_ids, inputs.attention_mask, inputs.input_ids.shape[1])
                    
                    all_old_log_probs.append(old_log_probs)
                    all_ref_log_probs.append(ref_log_probs)
                    
                    # Action mask (focus on routing decision)
                    action_mask = torch.ones_like(inputs.input_ids)
                    all_action_masks.append(action_mask)
                    
                    # Store final answer for each supervisor response
                    all_final_answers.append(final_answer)
                    all_repeated_prompts.append(sample["prompt"])
                    all_repeated_answers.append(sample["answer"])
    
    # Stack all tensors
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    old_log_probs = torch.cat(all_old_log_probs, dim=0)
    ref_log_probs = torch.cat(all_ref_log_probs, dim=0)
    action_mask = torch.cat(all_action_masks, dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "supervisor_responses": all_supervisor_responses,
        "final_answers": all_final_answers,
        "repeated_prompts": all_repeated_prompts,
        "repeated_answers": all_repeated_answers,
        "batch_size": len(batch_samples),
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
    
    # Handle both old and new reward function formats
    if "final_answers" in rollout_data:
        # Chained format
        rewards = torch.tensor(
            reward_function(
                final_answers=rollout_data["final_answers"],

            ),
            dtype=torch.float32,
            device=device
        )
    else:
        # Original format
        rewards = torch.tensor(
            reward_function(
                prompts=rollout_data["repeated_prompts"], 
                completions=rollout_data["formatted_completions"], # subagent outputs
                answer=rollout_data["repeated_answers"],
                supervisor_responses=rollout_data.get("supervisor_responses", None)  # supervisor responses for token penalty
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
    num_steps=5,
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

            # 1️⃣ Generate chained rollouts
            rollout_data = generate_chained_rollout_data(
                model,
                ref_model,
                tokenizer,
                batch_samples,
                num_generations,
                max_completion_length,
                max_steps=3
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
                "avg_reward": avg_reward
            }
            if "loss" in locals():
                log_data["loss"] = loss.item()

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
def evaluate_supervisor_chained(model, tokenizer, eval_examples, device=None, max_supervisor_new_tokens=64, max_steps=5):
    """Evaluates accuracy with chained multi-agent system."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct, total = 0, len(eval_examples)
    print("\n" + "="*50)
    print(f"STARTING CHAINED SUPERVISOR EVALUATION ON {total} EXAMPLES")
    print(f"Max steps per example: {max_steps}")
    print("="*50)

    for ex in eval_examples:
        print(f"\n--- Processing Example ---")
        expected = ex["answer"]
        conversation_history = []
        final_pred = None
        
        for step in range(max_steps):
            print(f"\n[STEP {step + 1}/{max_steps}]")
            
            # 1) Build supervisor prompt with conversation history
            sup_prompt = build_supervisor_prompt(ex, conversation_history)
            print(f"[DEBUG] Supervisor prompt:\n{sup_prompt}")
            print(f"[DEBUG] Prompt length: {len(sup_prompt)} characters")
            
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
            
            # Extract only the newly generated tokens
            input_length = inputs.shape[1]
            new_tokens = outputs[0][input_length:]
            sup_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up supervisor response - remove any text before "Agent:"
            if "Agent:" in sup_response:
                sup_response = sup_response[sup_response.find("Agent:"):]
            
            # 3) Parse agent choice
            agent = parse_supervisor_choice(sup_response)
            print(f"[DEBUG] Supervisor response: {sup_response}")
            print(f"[DEBUG] Parsed agent choice: {agent}")
            
            # 4) Force answering agent if max_steps reached
            if step == max_steps - 1:
                agent = "answering"
                print(f"[DEBUG] Max steps reached, forcing answering agent")
            
            # 5) Run sub-agent with conversation history
            if agent in VALID_AGENTS:
                print(f"[DEBUG] Calling sub-agent: {agent}")
                sub_text, pred = run_subagent(agent, ex, conversation_history)
                print(f"[DEBUG] Sub-agent response: {sub_text}...")
                print(f"[DEBUG] Extracted answer: {pred}")
                
                # Add supervisor decision and sub-agent output to conversation history
                clean_decision = re.sub(r"(?im)^.*Agent:\s*", "", sup_response).strip()
                conversation_history.append(("supervisor", f"Chose {agent} ({clean_decision})"))
                conversation_history.append((agent, sub_text))
                
                # If this is the answering agent, we're done
                if agent == "answering":
                    final_pred = pred
                    print(f"[DEBUG] Answering agent called, final answer: {pred}")
                    break
            else:
                print(f"[WARNING] Invalid agent '{agent}', skipping")
                # If invalid agent and max steps reached, default to answering
                if step == max_steps - 1:
                    agent = "answering"
                    sub_text, pred = run_subagent(agent, ex, conversation_history)
                    conversation_history.append(("supervisor", f"Chose {agent}: {sup_response}"))
                    conversation_history.append((agent, sub_text))
                    final_pred = pred
                    break
        
        # 6) Score the final prediction
        is_correct = (final_pred == expected) if final_pred else False
        correct += int(is_correct)
        
        # Logging
        print(f"\n--- Final Result ---")
        print(f"Expected: {expected} | Final Pred: {final_pred} | Correct: {'✓' if is_correct else '✗'}")
        print(f"Conversation history: {len(conversation_history)} steps")
        for i, (agent, output) in enumerate(conversation_history):
            print(f"  {i+1}. {agent}: {output[:50]}...")
        print("-"*50)

    acc = 100.0 * correct / max(total, 1)
    print(f"\nChained Evaluation Complete. Accuracy: {acc:.2f}% ({correct}/{total})")
    wandb.log({"chained_evaluation_accuracy": acc})

    print("="*50)
    model.train()
    return acc

def evaluate_supervisor(model, tokenizer, eval_examples, device=None, max_supervisor_new_tokens=64):
    """Evaluates accuracy by: supervisor routes -> sub-agent answers -> compare to gold yes/no."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct, total = 0, len(eval_examples)
    print("\n" + "="*50)
    print(f"STARTING SUPERVISOR EVALUATION ON {total} EXAMPLES")
    print("="*50)

    for ex in eval_examples:
        # 1) Build supervisor prompt from the original user prompt
        sup_prompt = build_supervisor_prompt(ex, conversation_history=None)
        print(f"[DEBUG] Supervisor prompt:\n{sup_prompt}")
        print(f"[DEBUG] Prompt length: {len(sup_prompt)} characters")
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
        # Extract only the newly generated tokens (not the input prompt)
        input_length = inputs.shape[1]
        new_tokens = outputs[0][input_length:]
        sup_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 3) Parse agent choice (default to 'answering' if bad format)
        agent = parse_supervisor_choice(sup_response)
        print(f"\n[DEBUG] Parsed agent choice: {agent}")
        print(f"[DEBUG] Valid agents: {VALID_AGENTS}")
        print(f"[DEBUG] Agent in valid agents: {agent in VALID_AGENTS}")

        # 4) Run sub-agent via Ollama on the ORIGINAL user prompt
        print(f"\n[DEBUG] About to call sub-agent: {agent}")
        if agent in VALID_AGENTS:
            print(f"[DEBUG] Agent is valid, calling run_subagent...")
            sub_text, pred = run_subagent(agent, ex, conversation_history=None)
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
