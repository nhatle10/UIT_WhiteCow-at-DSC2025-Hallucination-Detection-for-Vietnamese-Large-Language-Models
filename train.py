import unsloth  # noqa: F401 — side-effect import required by unsloth internals
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import argparse
import os
import shutil
import time
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from utils import (
    load_fewshot_data,
    sample_fewshots,
    free_gpu,
    build_user_msg,
    build_assistant_msg,
    BoolMaskCollator,
)

# Maps chat_template name → (instruction_part, response_part) tokens for train_on_responses_only
CHAT_TEMPLATE_CONFIG = {
    "gemma-3": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "qwen3-instruct": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


# ==========================
# Dataset Building
# ==========================

def build_train_dataset(train_csv, tokenizer, fewshot_data, smoke=False):
    df = pd.read_csv(train_csv)

    if smoke:
        df = df.head(5)

    required = ["id", "context", "prompt", "response", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    def row_to_conv(row):
        fewshot_subset = sample_fewshots(fewshot_data, k=5, seed=str(row["id"]))
        user_msg = build_user_msg(
            row["context"], row["prompt"], row["response"], fewshot_subset
        )
        assistant_msg = build_assistant_msg(row["label"])
        return {
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    conv_ds = Dataset.from_pandas(df).map(row_to_conv)

    def to_text(examples):
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in examples["conversations"]
        ]
        return {"text": texts}

    text_ds = conv_ds.map(to_text, batched=True)
    print(f"==> Dataset processed: {len(text_ds)} samples")
    return text_ds



# ==========================
# Training Function
# ==========================

def _load_base_model_with_peft(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=not args.load_in_8bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def train(args):
    free_gpu()

    print(f"==> Using {'8' if args.load_in_8bit else '4'}-bit quantization")

    fewshot_data = load_fewshot_data(args.fewshot_path)

    if args.mode == "continue_train":
        checkpoint_path = args.train_from or args.out_dir
        print(f"==> Continue training from: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=args.max_seq_len,
                load_in_4bit=not args.load_in_8bit,
                load_in_8bit=args.load_in_8bit,
                full_finetuning=False,
            )
            print(f"==> Loaded LoRA checkpoint from {checkpoint_path}")
        else:
            print("[WARNING] Checkpoint not found, starting fresh training")
            model, tokenizer = _load_base_model_with_peft(args)
    else:
        print("==> Fresh training mode")
        print(f"==> Loading base model: {args.model_name}")
        model, tokenizer = _load_base_model_with_peft(args)

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    text_ds = build_train_dataset(args.train_csv, tokenizer, fewshot_data, args.smoke)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=text_ds,
        data_collator=BoolMaskCollator(tokenizer),
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr * 0.5 if args.mode == "continue_train" else args.lr,
            logging_steps=args.logging_steps,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            output_dir=args.out_dir,
        ),
    )

    instruction_part, response_part = CHAT_TEMPLATE_CONFIG[args.chat_template]
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    print("==> Starting training...")
    stats = trainer.train()
    print("==> Train done:", stats.metrics.get("train_runtime"))

    print(f"==> Saving LoRA model to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    vllm_path = f"{args.out_dir}_vllm"
    print(f"==> Saving merged model for VLLM to {vllm_path}")
    model.save_pretrained_merged(vllm_path, tokenizer, save_method="merged_16bit")

    print("==> Training completed successfully!")


# ==========================
# Main
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detection model")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "continue_train"],
                        help="Training mode: fresh training or continue from checkpoint")

    # Data paths
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--fewshot_path", type=str, default="data/few_shot.json",
                        help="Path to few-shot examples JSON")
    parser.add_argument("--train_from", type=str,
                        help="Checkpoint path to continue training from")
    parser.add_argument("--out_dir", type=str, default="lora_model",
                        help="Output directory for model")

    # Model config
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name (e.g. unsloth/gemma-3-4b-it or unsloth/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--chat_template", type=str, required=True,
                        choices=list(CHAT_TEMPLATE_CONFIG.keys()),
                        help="Chat template to use")
    parser.add_argument("--max_seq_len", type=int, default=8096,
                        help="Maximum sequence length")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Use 8-bit quantization (default: 4-bit)")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")

    # Training config
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint frequency")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")

    # Debug
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: use only 5 samples to verify the pipeline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "continue_train" and os.path.exists(args.out_dir):
        backup_dir = f"{args.out_dir}_backup_{int(time.time())}"
        shutil.copytree(args.out_dir, backup_dir)
        print(f"==> Backup created at: {backup_dir}")

    train(args)
