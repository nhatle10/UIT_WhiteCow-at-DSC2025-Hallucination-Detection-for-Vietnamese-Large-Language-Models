# UIT_DSC 2025 - Hallucination Detection

## Overview

This project detects hallucinations in Vietnamese Large Language Models (LLMs). The system classifies LLM-generated responses into three categories based on their consistency with the provided context and prompt:

- **No Hallucination (no)**: The response is fully consistent with the information in the passage. It does not contain any unsupported information. It correctly answers the prompt based only on the provided context.
- **Intrinsic Hallucination (intrinsic)**: The response contradicts or distorts information specifically mentioned in the passage. The model misinterprets entities, numbers, or relationships present in the source.
- **Extrinsic Hallucination (extrinsic)**: The response contains additional information not found in the passage. Crucially, even if the information is factually true in the real world (e.g., general knowledge), if it cannot be derived from the passage, it is classified as extrinsic.

![alt text](assets/task_illustration.png)

## Project Structure

```
├── train_gemma3.py       # Training script for Gemma-3-4B model
├── train_qwen3.py        # Training script for Qwen3-4B model
├── inference.py          # Inference script with multiple modes
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
└── data/
    ├── few_shot.json     # Few-shot examples
    ├── train/            # Training data
    ├── test/             # Test data
    └── warmup/           # Warmup data
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU
- 50GB+ free disk space

### Setup

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **unsloth**: Optimized LLM training with LoRA
- **transformers**: HuggingFace Transformers (v4.55.4)
- **trl**: Trainer library (v0.22.2)
- **vllm**: Fast inference engine
- **peft**: Parameter-Efficient Fine-Tuning
- **bitsandbytes**: 4-bit/8-bit quantization
- **accelerate**: Training acceleration

## Data Format

### Training/Test CSV

Required columns:

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier |
| `context` | Reference context |
| `prompt` | Question/request |
| `response` | LLM response |
| `label` | Label: no/intrinsic/extrinsic (training only) |

### Few-shot Examples

The `few_shot.json` file contains example samples used as few-shot demonstrations during training and inference.

## Usage

### Training Models

#### Gemma-3-4B (Fresh Training)

```bash
python train_gemma3.py \
  --mode train \
  --train_csv data/train/vihallu-train.csv \
  --fewshot_path data/few_shot.json \
  --out_dir lora_gemma3 \
  --model_name unsloth/gemma-3-4b-it \
  --max_seq_len 8096 \
  --lora_r 32 \
  --lora_alpha 32 \
  --epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --weight_decay 0.01
```

#### Qwen3-4B (Fresh Training)

```bash
python train_qwen3.py \
  --mode train \
  --train_csv data/train/vihallu-train.csv \
  --fewshot_path data/few_shot.json \
  --out_dir lora_qwen3 \
  --model_name unsloth/Qwen2.5-4B-Instruct \
  --max_seq_len 8096 \
  --lora_r 32 \
  --lora_alpha 32 \
  --epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --weight_decay 0.01
```

#### Continue Training from Checkpoint

```bash
python train_gemma3.py \
  --mode continue_train \
  --train_csv data/train/vihallu-train.csv \
  --fewshot_path data/few_shot.json \
  --out_dir lora_gemma3
```

### Inference

#### Standard Inference

```bash
python inference.py \
  --mode vllm \
  --test_csv data/test/vihallu-public-test.csv \
  --fewshot_path data/few_shot.json \
  --out_csv predictions.csv \
  --base_model unsloth/gemma-3-4b-it \
  --lora_dir lora_gemma3 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --max_new_tokens 512
```

#### Multi-temperature Inference

```bash
python inference.py \
  --mode multi_temp \
  --test_csv data/test/vihallu-public-test.csv \
  --fewshot_path data/few_shot.json \
  --base_model unsloth/gemma-3-4b-it \
  --lora_dir lora_gemma3 \
  --temperatures 0.5,0.7,0.9 \
  --raw_output_dir outputs/ \
  --model_prefix gemma3
```

#### Ensemble Inference

Combine predictions from multiple models and temperature settings.

```bash
python inference.py \
  --mode ensemble \
  --ensemble_configs config.json \
  --test_csv data/test/vihallu-public-test.csv \
  --out_csv ensemble_predictions.csv \
  --voting_method majority
```

## Training Parameters

### Key Arguments

- `--model_name`: Base model identifier (e.g., `unsloth/gemma-3-4b-it`)
- `--max_seq_len`: Maximum sequence length (default: 8096)
- `--lora_r`: LoRA rank (default: 32)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--lr`: Learning rate (default: 5e-5)
- `--weight_decay`: Weight decay for regularization
- `--load_in_8bit`: Use 8-bit quantization (default: 4-bit)

## Output

Training outputs:
- LoRA adapter weights in `{out_dir}/`
- GGUF format model (if conversion enabled)
- vLLM-compatible model in `{out_dir}_vllm/`

Inference outputs:
- CSV file with columns: `id`, `predict_label`, `raw_output`
- Label distribution statistics

## Key Functions (utils.py)

- `load_fewshot_data()`: Load few-shot examples from JSON
- `sample_fewshots()`: Sample k few-shot examples
- `build_user_msg()`: Format user message with context and prompt
- `build_assistant_msg()`: Format assistant response with label
- `BoolMaskCollator`: Custom collator for attention mask conversion
- `free_gpu()`: Clear GPU memory

## Notes

- Models are trained using LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Few-shot prompting is used during both training and inference
- Inference uses vLLM for fast batch processing
- Supports 4-bit and 8-bit quantization for reduced memory usage

## File Descriptions

- `train_gemma3.py`: Handles Gemma-3-4B model training and conversion
- `train_qwen3.py`: Handles Qwen3-4B model training and conversion
- `inference.py`: Implements inference with vLLM, multi-temperature, and ensemble modes
- `utils.py`: Shared utility functions for data loading, processing, and messaging
- `requirements.txt`: Python package dependencies

## References

- [Unsloth](https://github.com/unslothai/unsloth): Fast language model training
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [vLLM](https://docs.vllm.ai/): Fast inference engine
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09714)
