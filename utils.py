import json
import random
import torch
from collections import defaultdict
from transformers import DataCollatorForLanguageModeling

# ==========================
# Data Loading
# ==========================

def load_fewshot_data(fewshot_json_path="data/few_shot.json"):
    """
    Load few-shot examples from a JSON file.

    Args:
        fewshot_json_path (str): Path to the few-shot examples JSON file.

    Returns:
        list[dict]: List of few-shot example dictionaries.
    """
    with open(fewshot_json_path, "r") as f:
        fewshot_data = json.load(f)
    return fewshot_data[:]


LABELS = ["no", "intrinsic", "extrinsic"]


# ==========================
# Custom Collator
# ==========================

class BoolMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "attention_mask" in batch:
            mask = batch["attention_mask"]
            if mask.dtype != torch.bool:
                batch["attention_mask"] = mask.bool()
        return batch


# ==========================
# Few-shot Sampling
# ==========================

def sample_fewshots(fewshot_data, k=5, seed=None):
    """
    Sample k few-shot examples ensuring each label is represented at least once.

    Logic:
        1. Guarantee one sample per label (no, intrinsic, extrinsic)
        2. Randomly sample remaining (k - 3) examples from the rest
        3. Shuffle the final order to avoid position bias

    Args:
        fewshot_data (list[dict]): List of few-shot example dictionaries.
        k (int): Number of few-shot examples to sample (must be >= number of labels).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list[dict]: Sampled few-shot examples.
    """
    if k < len(LABELS):
        raise ValueError(f"k={k} must be >= number of labels {len(LABELS)}")

    # Group examples by label
    by_label = defaultdict(list)
    for ex in fewshot_data:
        lab = ex.get("label")
        if lab in LABELS:
            by_label[lab].append(ex)

    # Ensure each label has at least one example
    missing = [lab for lab in LABELS if len(by_label[lab]) == 0]
    if missing:
        raise ValueError(f"Missing few-shot examples for labels: {missing}")

    rng = random.Random(seed)

    selected = [rng.choice(by_label[lab]) for lab in LABELS]

    used_ids = set(map(id, selected))
    remaining_pool = [ex for ex in fewshot_data if id(ex) not in used_ids]

    if len(remaining_pool) < (k - len(LABELS)):
        raise ValueError("Not enough few-shot examples to sample the requested number.")

    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[:(k - len(LABELS))])

    rng.shuffle(selected)
    return selected


# ==========================
# GPU Memory Management
# ==========================

def free_gpu():
    """
    Forcefully clear and synchronize GPU memory.

    This function is useful when performing multiple sequential model runs
    to avoid OOM (Out of Memory) errors.

    Actions:
        - Garbage collect Python objects
        - Empty CUDA cache
        - Synchronize devices
        - Reset CUDA memory stats
    """
    import gc
    import torch

    # Force garbage collection multiple times
    for _ in range(5):
        gc.collect()

    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Force clear all devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Memory info
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[INFO] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        )

    print("[INFO] GPU memory cleaned!")


# ==========================
# Text Processing
# ==========================

def normalize_label(s: str) -> str:
    """
    Normalize input string into one of the canonical labels: no | intrinsic | extrinsic.

    Args:
        s (str): Raw label string from dataset or model output.

    Returns:
        str: Normalized label ('no', 'intrinsic', or 'extrinsic').
    """
    s = (s or "").strip().lower()
    for lab in LABELS:
        if s.startswith(lab) or lab in s:
            return lab
    return "no"


# ==========================
# Prompt Building
# ==========================

def build_few_shots(fewshot_data):
    """
    Construct formatted few-shot examples as text prompt.

    Args:
        fewshot_data (list[dict]): List of few-shot examples, each containing
            'context', 'prompt', 'generated_response', 'label', and 'explanation'.

    Returns:
        str: Concatenated few-shot examples in readable format.
    """
    FEWSHOT = "Context: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\n\nLabel: {label}\n\nExplanation: {explanation}\n\n"
    result = ""
    for data in fewshot_data:
        result += FEWSHOT.format(
            context=data["context"],
            prompt=data["prompt"],
            response=data["generated_response"],
            label=data["label"],
            explanation=data["explanation"],
        )
    return result


def build_user_msg(context, prompt, response, fewshot_data):
    """
    Build the full instruction + few-shot + target classification input for the LLM.

    Args:
        context (str): Supporting context passage.
        prompt (str): User query or model prompt.
        response (str): Model-generated response to classify.
        fewshot_data (list[dict]): Few-shot demonstration examples.

    Returns:
        str: A single concatenated instruction message for the classifier model.
    """
    INSTRUCTION = """You are a hallucination detection classifier for Vietnamese language models. 
Your task is to classify the RESPONSE into exactly ONE label from {no, intrinsic, extrinsic}, 
based ONLY on the given CONTEXT and PROMPT. 
You must NEVER use knowledge outside the provided CONTEXT.

Label Definitions:
- no: RESPONSE is fully supported by CONTEXT, with no added or fabricated content. 
       Allowed to reject false assumptions in PROMPT if CONTEXT shows they are wrong.
- intrinsic: RESPONSE contradicts, reverses, or distorts facts from CONTEXT. 
             This includes repeating false assumptions from PROMPT that conflict with CONTEXT.
- extrinsic: RESPONSE adds new information not grounded in CONTEXT and not directly verifiable from it, 
             without explicit contradiction.

Classification Rules:
1) If RESPONSE both contradicts CONTEXT AND adds unsupported info → intrinsic (contradiction takes priority).
2) Match at semantic level; ignore minor spelling or grammatical errors.
3) If PROMPT contains false assumptions and RESPONSE accepts/repeats them against CONTEXT → intrinsic.
4) If RESPONSE only says "insufficient / not enough information" (without fabricating) → no.
5) Output must be EXACTLY one word: no | intrinsic | extrinsic

Evaluation Order:
1. First check for contradictions with CONTEXT → intrinsic
2. If no contradiction, check for unsupported additions → extrinsic
3. If fully supported with no addition → no
"""

    FEWSHOT = """EXAMPLE CLASSIFICATION:\n\n\n""" + build_few_shots(fewshot_data)

    return (
        INSTRUCTION
        + "\n\n"
        + FEWSHOT
        + "\n\n"
        + f"Please classify the following samples:\n\nContext: {context}\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response: {response}\n"
        f"Label:"
    )


def build_assistant_msg(label):
    """Builds assistant's reply message with normalized label only."""
    return f"{normalize_label(label)}"