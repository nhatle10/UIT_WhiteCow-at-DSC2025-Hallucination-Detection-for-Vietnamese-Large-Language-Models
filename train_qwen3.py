import unsloth
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



# ==========================
# Dataset Building
# ==========================

def build_train_dataset(train_csv, tokenizer, fewshot_data, mock=False):
    """Build training dataset from CSV file"""
    df = pd.read_csv(train_csv)

    if mock:
        df = df.head(5)

    required = ["id", "context", "prompt", "response", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col}")

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

    ds = Dataset.from_pandas(df).map(row_to_conv)

    def to_text(examples):
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in examples["conversations"]
        ]
        return {"text": texts}

    final_ds = ds.map(to_text, batched=True)
    print(f"==> Dataset processed: {len(final_ds)} samples")
    return final_ds


# ==========================
# Training Function
# ==========================

def train(args):
    """Main training function"""
    # GPU cleanup
    free_gpu()

    if args.load_in_8bit:
        print("==> Using 8-bit quantization")
    else:
        print("==> Using 4-bit quantization")

    # Load fewshot data
    fewshot_data = load_fewshot_data(args.fewshot_path)

    # Continue training mode
    if args.mode == "continue_train":
        # Determine checkpoint path to load
        if args.train_from:
            checkpoint_path = args.train_from
            print(f"==> Continue training from specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.out_dir
            print(f"==> Continue training from default checkpoint: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            # Load from saved LoRA checkpoint
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=args.max_seq_len,
                load_in_4bit=not args.load_in_8bit,
                load_in_8bit=args.load_in_8bit,
                full_finetuning=False,
            )
            print(f"==> Successfully loaded LoRA checkpoint from {checkpoint_path}")
        else:
            print(
                f"WARNING: Checkpoint {checkpoint_path} not found, starting fresh training"
            )
            print(f"==> Loading base model: {args.model_name}")
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
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )

    # Fresh training mode
    else:
        print("==> Fresh training mode")
        print(f"==> Loading base model: {args.model_name}")
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
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )

    # Load train dataset
    mock = bool(args.mock)
    train_ds = build_train_dataset(args.train_csv, tokenizer, fewshot_data, mock)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None,
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

    # Train on responses only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Start training
    print("==> Starting training...")
    stats = trainer.train()
    print("==> Train done:", stats.metrics.get("train_runtime"))

    # Save model (LoRA adapters)
    print(f"==> Saving LoRA model to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Save merged model for VLLM
    vllm_path = f"{args.out_dir}_vllm"
    print(f"==> Saving merged model for VLLM to {vllm_path}")
    model.save_pretrained_merged(
        vllm_path, tokenizer, save_method="merged_16bit"
    )
    
    print("==> Training completed successfully!")


# ==========================
# Main
# ==========================

def parse_args():
    p = argparse.ArgumentParser(description="Train hallucination detection model")
    
    # Mode
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "continue_train"],
        help="Training mode: fresh training or continue from checkpoint"
    )
    
    # Data paths
    p.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    p.add_argument("--fewshot_path", type=str, default="data/few_shot_2.json", 
                   help="Path to few-shot examples JSON")
    p.add_argument("--train_from", type=str, help="Checkpoint path to continue training from")
    p.add_argument("--out_dir", type=str, default="lora_model", help="Output directory for model")
    
    # Model config
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507",
                   help="Base model name")
    p.add_argument("--max_seq_len", type=int, default=8096, help="Maximum sequence length")
    p.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    
    # LoRA config
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")
    
    # Training config
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=4, 
                   help="Batch size per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    p.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint frequency")
    p.add_argument("--seed", type=int, default=3407, help="Random seed")
    
    # Debug
    p.add_argument("--mock", type=int, default=0, help="Use only 5 samples for testing")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Backup existing model if continuing training
    if args.mode == "continue_train" and os.path.exists(args.out_dir):
        backup_dir = f"{args.out_dir}_backup_{int(time.time())}"
        shutil.copytree(args.out_dir, backup_dir)
        print(f"==> Backup created at: {backup_dir}")
    
    # Run training
    train(args)