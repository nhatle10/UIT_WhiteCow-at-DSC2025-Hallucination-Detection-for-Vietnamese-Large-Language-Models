import json
import random
import torch
from collections import defaultdict
from transformers import DataCollatorForLanguageModeling

# ==========================
# Data Loading
# ==========================

def load_fewshot_data(fewshot_json_path="data/few_shot.json"):
    with open(fewshot_json_path, "r") as f:
        return json.load(f)


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
    if k < len(LABELS):
        raise ValueError(f"k={k} must be >= number of labels {len(LABELS)}")

    by_label = defaultdict(list)
    for example in fewshot_data:
        label = example.get("label")
        if label in LABELS:
            by_label[label].append(example)

    missing = [label for label in LABELS if not by_label[label]]
    if missing:
        raise ValueError(f"Missing few-shot examples for labels: {missing}")

    rng = random.Random(seed)

    selected = [rng.choice(by_label[label]) for label in LABELS]

    # Use Python object id to avoid duplicating already-selected examples
    selected_object_ids = set(map(id, selected))
    remaining_pool = [ex for ex in fewshot_data if id(ex) not in selected_object_ids]

    if len(remaining_pool) < (k - len(LABELS)):
        raise ValueError("Not enough few-shot examples to sample the requested number.")

    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[: (k - len(LABELS))])

    rng.shuffle(selected)
    return selected


# ==========================
# GPU Memory Management
# ==========================

def free_gpu():
    import gc

    for _ in range(5):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"[INFO] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    print("[INFO] GPU memory cleaned!")


# ==========================
# Text Processing
# ==========================

def normalize_label(raw_label: str) -> str:
    raw_label = (raw_label or "").strip().lower()
    for label in LABELS:
        if label in raw_label:
            return label
    return "no"


# ==========================
# Prompt Building
# ==========================

_FEWSHOT_TEMPLATE = "Context: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\n\nLabel: {label}\n\nExplanation: {explanation}\n\n"

_INSTRUCTION = """You are a hallucination detection classifier for Vietnamese language models.
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


def build_few_shots(fewshot_data):
    return "".join(
        _FEWSHOT_TEMPLATE.format(
            context=example["context"],
            prompt=example["prompt"],
            response=example["generated_response"],
            label=example["label"],
            explanation=example["explanation"],
        )
        for example in fewshot_data
    )


def build_user_msg(context, prompt, response, fewshot_data):
    few_shot_section = "EXAMPLE CLASSIFICATION:\n\n\n" + build_few_shots(fewshot_data)

    return (
        _INSTRUCTION
        + "\n\n"
        + few_shot_section
        + "\n\n"
        + f"Please classify the following samples:\n\nContext: {context}\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response: {response}\n"
        f"Label:"
    )


def build_assistant_msg(label):
    return normalize_label(label)