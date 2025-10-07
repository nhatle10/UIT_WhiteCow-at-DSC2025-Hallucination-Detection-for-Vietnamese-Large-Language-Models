import json
import random
import torch
from collections import defaultdict
from transformers import DataCollatorForLanguageModeling


# ==========================
# Data Loading
# ==========================

def load_fewshot_data(fewshot_json_path="data/few_shot_2.json"):
    """Load few-shot examples from JSON file"""
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
    Lấy k ví dụ few-shot với điều kiện:
      - có đủ cả 3 nhãn trong LABELS (mỗi nhãn >= 1)
      - phần còn lại chọn ngẫu nhiên từ toàn bộ pool (trừ các mẫu đã lấy)
      - trộn ngẫu nhiên thứ tự kết quả
    """
    if k < len(LABELS):
        raise ValueError(f"k={k} phải >= số nhãn {len(LABELS)}")

    # Gom theo nhãn
    by_label = defaultdict(list)
    for ex in fewshot_data:
        lab = ex.get("label")
        if lab in LABELS:
            by_label[lab].append(ex)

    # Kiểm tra đủ nguồn mỗi nhãn
    missing = [lab for lab in LABELS if len(by_label[lab]) == 0]
    if missing:
        raise ValueError(f"fewshot_data thiếu ví dụ cho các nhãn: {missing}")

    rng = random.Random(seed)

    # Bước 1: đảm bảo phủ đủ 3 nhãn (mỗi nhãn chọn 1)
    selected = [rng.choice(by_label[lab]) for lab in LABELS]

    # Bước 2: chọn thêm (k-3) mẫu ngẫu nhiên từ phần còn lại
    used_ids = set(map(id, selected))
    remaining_pool = [ex for ex in fewshot_data if id(ex) not in used_ids]

    if len(remaining_pool) < (k - len(LABELS)):
        raise ValueError(f"Không đủ few-shots để lấy {k} mẫu (pool còn {len(remaining_pool)})")

    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[:(k - len(LABELS))])

    # Bước 3: trộn thứ tự kết quả để tránh bias vị trí
    rng.shuffle(selected)
    return selected


# ==========================
# GPU Memory Management
# ==========================

def free_gpu():
    """Enhanced GPU memory cleanup"""
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

def norm(x):
    return "" if x is None or str(x).lower() == "nan" else str(x)


def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    for lab in ["extrinsic", "intrinsic", "no"]:
        if s.startswith(lab) or lab in s:
            return lab
    return "no"


# ==========================
# Prompt Building
# ==========================

def build_few_shots(fewshot_data):
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
    return f"{normalize_label(label)}"