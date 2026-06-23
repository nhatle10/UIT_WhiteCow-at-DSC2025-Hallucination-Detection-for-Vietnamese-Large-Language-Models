import argparse
import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter

from utils import (
    load_fewshot_data,
    build_user_msg,
    normalize_label,
    free_gpu,
)


# ==========================
# Inference Function
# ==========================

def _resolve_model_path(args):
    if args.model_path:
        print(f"==> Using specified model: {args.model_path}")
        return args.model_path

    vllm_model_path = f"{args.lora_dir}_vllm"
    if os.path.exists(vllm_model_path):
        print(f"==> Using fine-tuned model: {vllm_model_path}")
        return vllm_model_path

    print(f"==> Fine-tuned model not found. Using base model: {args.base_model}")
    return args.base_model


def inference_vllm(args):
    print("==> Starting inference...")
    free_gpu()

    fewshot_data = load_fewshot_data(args.fewshot_path)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    print(f"==> Sampling params: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

    model_path = _resolve_model_path(args)
    print(f"==> Loading model from {model_path}")
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=args.max_seq_len,
        gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"==> Loading test data from {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    if args.smoke:
        df = df.head(10)
        print(f"==> Smoke test: using only {len(df)} samples")

    required = ["id", "context", "prompt", "response"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    preds, raw_outputs = [], []
    print(f"==> Processing {len(df)} samples in {(len(df) + args.batch_size - 1) // args.batch_size} batches...")

    for start in tqdm(range(0, len(df), args.batch_size), desc="Inference"):
        batch = df.iloc[start : start + args.batch_size]

        prompts = []
        for _, row in batch.iterrows():
            user_msg = build_user_msg(
                row["context"],
                row["prompt"],
                row["response"],
                fewshot_data[:args.num_fewshots],
            )
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(input_text)

        for output in model.generate(prompts, sampling_params):
            decoded = output.outputs[0].text
            raw_outputs.append(decoded)
            preds.append(normalize_label(decoded))

    df["predict_label"] = preds
    df["raw_output"] = raw_outputs
    output_df = df[["id", "predict_label", "raw_output"]]
    output_df.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"==> Inference completed! Results saved to {args.out_csv}")
    print("\n==> Label distribution:")
    for label, count in output_df["predict_label"].value_counts().items():
        print(f"  {label}: {count} ({count / len(output_df) * 100:.1f}%)")


# ==========================
# Multi-temperature Inference
# ==========================

def inference_multi_temp(args):
    print("==> Running multi-temperature inference...")
    os.makedirs(args.raw_output_dir, exist_ok=True)

    temps = [float(t) for t in args.temperatures.split(",")]
    print(f"==> Temperatures: {temps}")

    output_files = []
    for temp in temps:
        out_file = os.path.join(args.raw_output_dir, f"{args.model_prefix}_temp_{temp}.csv")
        print(f"\n>>> Running with temp = {temp}")
        args.temperature = temp
        args.out_csv = out_file
        inference_vllm(args)
        output_files.append(out_file)

    print(f"\n==> Generated {len(output_files)} files with different temperatures")
    return output_files


# ==========================
# Post-processing
# ==========================

def process_outputs(raw_dir, save_dir):
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw output directory not found: {raw_dir}")
        print(f"        Please run with --mode multi_temp first to generate predictions")
        return []

    os.makedirs(save_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {raw_dir}")
        return []

    print(f"==> Found {len(csv_files)} files to process")

    processed_files = []
    for file_name in csv_files:
        df = pd.read_csv(os.path.join(raw_dir, file_name))
        out_path = os.path.join(save_dir, file_name.replace("temp_", "submit_"))
        df[["id", "predict_label"]].to_csv(out_path, index=False)
        processed_files.append(out_path)
        print(f"==> Processed: {file_name}")

    print(f"\n==> Processed {len(processed_files)} files successfully")
    print(f"==> Output directory: {save_dir}")
    return processed_files


def majority_voting(input_dir, output_file, filter_prefix=None):
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        print(f"        Please run with --mode process first to create processed outputs")
        return

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if filter_prefix:
        csv_files = [f for f in csv_files if filter_prefix in f]
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {input_dir}")
        return

    print(f"==> Loading {len(csv_files)} files for voting...")

    pred_dfs = []
    for file in csv_files:
        path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(path)
            if {"id", "predict_label"}.issubset(df.columns):
                pred_dfs.append(df)
                print(f"==> Loaded: {file}")
            else:
                print(f"[WARNING] Skipped (wrong format): {file}")
        except Exception as e:
            print(f"[WARNING] Error reading {file}: {e}")

    if not pred_dfs:
        print("[ERROR] No valid files for voting")
        return

    combined = pd.concat(pred_dfs, axis=0)

    final_rows = []
    for sample_id, group in combined.groupby("id"):
        voted_label = Counter(group["predict_label"]).most_common(1)[0][0]
        final_rows.append({"id": sample_id, "predict_label": voted_label})

    final_df = pd.DataFrame(final_rows).sort_values("id").reset_index(drop=True)
    final_df.to_csv(output_file, index=False)
    print(f"\n==> Saved voting result to: {output_file}")

    print("\n==> Final label distribution:")
    for label, count in final_df["predict_label"].value_counts().items():
        print(f"  {label}: {count}")


# ==========================
# Main
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for hallucination detection")

    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "multi_temp", "process", "voting"],
                        help="Inference mode")

    # Data paths
    parser.add_argument("--test_csv", type=str,
                        help="Path to test CSV file")
    parser.add_argument("--fewshot_path", type=str, default="data/few_shot.json",
                        help="Path to few-shot examples JSON")
    parser.add_argument("--out_csv", type=str,
                        help="Output CSV file path (for single mode)")

    # Model paths
    parser.add_argument("--model_path", type=str,
                        help="Direct path to model (overrides auto-detection)")
    parser.add_argument("--lora_dir", type=str, default="lora_model",
                        help="LoRA model directory (will look for {lora_dir}_vllm)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507",
                        help="Base model name (fallback if fine-tuned not found)")

    # Inference config
    parser.add_argument("--max_seq_len", type=int, default=8096,
                        help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num_fewshots", type=int, default=5,
                        help="Number of few-shot examples to use")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k sampling")

    # Multi-temperature mode
    parser.add_argument("--temperatures", type=str,
                        default="0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9",
                        help="Comma-separated temperatures for multi_temp mode")
    parser.add_argument("--model_prefix", type=str, default="model",
                        help="Prefix for output files in multi_temp mode")
    parser.add_argument("--raw_output_dir", type=str, default="raw_output",
                        help="Directory for raw outputs")

    # Post-processing
    parser.add_argument("--process_dir", type=str, default="process_output",
                        help="Directory for processed outputs")
    parser.add_argument("--voting_input_dir", type=str, default="process_output",
                        help="Input directory for voting")
    parser.add_argument("--voting_output", type=str, default="final_submit.csv",
                        help="Output file for voting result")
    parser.add_argument("--voting_filter", type=str,
                        help="Filter files by prefix for voting (e.g., 'qwen3')")

    # Debug
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: use only 10 samples to verify the pipeline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "single":
        if not args.test_csv or not args.out_csv:
            raise ValueError("--test_csv and --out_csv required for single mode")
        inference_vllm(args)

    elif args.mode == "multi_temp":
        if not args.test_csv:
            raise ValueError("--test_csv required for multi_temp mode")
        inference_multi_temp(args)

    elif args.mode == "process":
        print(f"==> Processing outputs from {args.raw_output_dir}")
        process_outputs(args.raw_output_dir, args.process_dir)

    elif args.mode == "voting":
        print(f"==> Running majority voting on {args.voting_input_dir}")
        majority_voting(args.voting_input_dir, args.voting_output, filter_prefix=args.voting_filter)
