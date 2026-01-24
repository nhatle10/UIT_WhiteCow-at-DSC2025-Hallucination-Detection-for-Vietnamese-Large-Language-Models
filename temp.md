# UIT_DSC 2025 - UIT_WhiteCow

## 📋 Tổng quan dự án

Dự án phát hiện hallucination (ảo giác) trong các mô hình ngôn ngữ lớn (LLM) tiếng Việt. Hệ thống phân loại các phản hồi của LLM thành 3 nhãn:

- **no**: Phản hồi được hỗ trợ đầy đủ bởi ngữ cảnh, không có nội dung được thêm vào hoặc bịa đặt
- **intrinsic**: Phản hồi mâu thuẫn, đảo ngược hoặc bóp méo thông tin từ ngữ cảnh
- **extrinsic**: Phản hồi thêm thông tin mới không có cơ sở trong ngữ cảnh

## 🏗️ Kiến trúc hệ thống

### Thành phần chính

```
UIT_DSC/
├── train_gemma3.py       # Script huấn luyện mô hình Gemma-3-4B
├── train_qwen3.py        # Script huấn luyện mô hình Qwen3-4B
├── inference.py          # Script inference với nhiều chế độ
├── utils.py              # Các hàm tiện ích
├── requirements.txt      # Dependencies
└── data/
    ├── few_shot.json     # Ví dụ few-shot 
    ├── train/            # Dữ liệu huấn luyện
    ├── test/             # Dữ liệu kiểm tra
    └── warmup/           # Dữ liệu warmup
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA-capable GPU 
- 50GB+ dung lượng ổ cứng trống

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Dependencies chính

- **unsloth**: Framework tối ưu hóa training LLM với LoRA
- **transformers**: Thư viện HuggingFace Transformers (v4.55.4)
- **trl**: Transformer Reinforcement Learning (v0.22.2)
- **vllm**: Inference engine tốc độ cao
- **bitsandbytes**: Quantization (4-bit/8-bit)
- **peft**: Parameter-Efficient Fine-Tuning
- **accelerate**: Tăng tốc training

## 📚 Cấu trúc dữ liệu

### Training CSV

Cần có các cột sau:

| Cột | Mô tả |
|-----|-------|
| `id` | ID duy nhất của mẫu |
| `context` | Ngữ cảnh tham khảo |
| `prompt` | Câu hỏi/yêu cầu |
| `response` | Phản hồi của LLM |
| `label` | Nhãn: no/intrinsic/extrinsic |

### Test CSV

Cần có các cột:

| Cột | Mô tả |
|-----|-------|
| `id` | ID duy nhất của mẫu |
| `context` | Ngữ cảnh tham khảo |
| `prompt` | Câu hỏi/yêu cầu |
| `response` | Phản hồi của LLM cần phân loại |


## 🎯 Huấn luyện mô hình

### 1. Huấn luyện Gemma-3-4B

#### Huấn luyện mới (Fresh training)

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
  --lora_dropout 0 \
  --epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 10000 \
  --seed 3407
```

#### Tiếp tục huấn luyện (Continue training)

```bash
python train_gemma3.py \
  --mode continue_train \
  --train_csv data/train/vihallu-train.csv \
  --fewshot_path data/few_shot.json \
  --out_dir lora_gemma3 \
  --epochs 1
```

#### Các tham số quan trọng

- `--mode`: Chế độ training (`train` hoặc `continue_train`)
- `--train_csv`: Đường dẫn file CSV training
- `--fewshot_path`: Đường dẫn file JSON few-shot examples
- `--out_dir`: Thư mục lưu model (sẽ tạo thêm `{out_dir}_vllm` cho inference)
- `--model_name`: Tên base model từ HuggingFace
- `--max_seq_len`: Độ dài sequence tối đa (mặc định: 8096)
- `--load_in_8bit`: Sử dụng quantization 8-bit thay vì 4-bit
- `--lora_r`: LoRA rank (mặc định: 32)
- `--lora_alpha`: LoRA alpha (mặc định: 32)
- `--lora_dropout`: LoRA dropout (mặc định: 0)
- `--epochs`: Số epoch huấn luyện
- `--per_device_train_batch_size`: Batch size trên mỗi GPU
- `--gradient_accumulation_steps`: Số bước tích lũy gradient
- `--lr`: Learning rate (mặc định: 5e-5)
- `--weight_decay`: Weight decay (mặc định: 0.01)
- `--mock`: Chế độ test với 5 mẫu đầu tiên

### 2. Huấn luyện Qwen3-4B

Tương tự Gemma-3, nhưng sử dụng `train_qwen3.py`:

```bash
python train_qwen3.py \
  --mode train \
  --train_csv data/train/vihallu-train.csv \
  --fewshot_path data/few_shot.json \
  --out_dir lora_qwen3 \
  --model_name unsloth/Qwen3-4B-Instruct-2507 \
  --epochs 1
```

### Sự khác biệt giữa Gemma-3 và Qwen3

| Tính năng | Gemma-3 | Qwen3 |
|-----------|---------|-------|
| Base model | `unsloth/gemma-3-4b-it` | `unsloth/Qwen3-4B-Instruct-2507` |
| Chat template | `gemma-3` | `qwen3-instruct` |
| Instruction part | `<start_of_turn>user\n` | `<|im_start|>user\n` |
| Response part | `<start_of_turn>model\n` | `<|im_start|>assistant\n` |

### Output của quá trình training

- **LoRA adapters**: Lưu tại `{out_dir}/`
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `tokenizer` files
  
- **Merged model cho vLLM**: Lưu tại `{out_dir}_vllm/`
  - Full merged model ở format 16-bit
  - Sẵn sàng để inference với vLLM

## 🔮 Inference

`inference.py`:

### Mô hình đã được train và lưu lên huggingface nhatle10/uit_qwen3_reason và nhatle10/uit_gemma3-4b-it


Chạy inference với nhiều giá trị temperature khác nhau:

```bash
# Step 1: Generate predictions với nhiều temperature
!python3 inference.py \
  --mode multi_temp \
  --model_path nhatle10/uit_qwen3_reason \
  --test_csv ./data/test/vihallu-private-test.csv \
  --model_prefix qwen3_preds \
  --raw_output_dir raw_output

!python3 inference.py \
  --mode multi_temp \
  --model_path nhatle10/uit_gemma3-4b-it \
  --test_csv ./data/test/vihallu-private-test.csv \
  --model_prefix gemma3_preds \
  --raw_output_dir raw_output

!python3 inference.py \
  --mode process \
  --raw_output_dir raw_output \
  --process_dir process_output

!python3 inference.py \
  --mode voting \
  --voting_input_dir process_output \
  --voting_output final_submit.csv
```

**Output**: File CSV cuối cùng `final_submit.csv` với 2 cột: `id`, `predict_label`

## 🎓 Kỹ thuật sử dụng

### 1. Parameter-Efficient Fine-Tuning (PEFT) với LoRA

- **LoRA rank**: 32
- **LoRA alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Gradient checkpointing**: Tiết kiệm memory

### 2. Quantization

- **4-bit** (mặc định): Tiết kiệm memory tối đa
- **8-bit** (optional): Cân bằng giữa memory và accuracy

### 3. Training Strategy

- **Optimizer**: AdamW 8-bit
- **Scheduler**: Cosine với warmup_ratio=0.03
- **Train on responses only**: Chỉ tính loss trên phần response

### 4. Inference Optimization

- **vLLM**: Engine inference song song tốc độ cao
- **Batch processing**: Xử lý nhiều mẫu cùng lúc
- **GPU memory utilization**: 0.8 (80% VRAM)

### 5. Ensemble Strategy

- **Multi-temperature sampling**: Thử nghiệm nhiều mức độ randomness
- **Majority voting**: Kết hợp dự đoán từ nhiều model/temperature

###  Chọn model

| Model | Ưu điểm | Nhược điểm |
|-------|---------|------------|
| Gemma-3-4B | Stable, dễ train | Kém hơn về tiếng Việt |
| Qwen3-4B | Tốt hơn cho tiếng Việt | Đôi khi overfit |

**Khuyến nghị**: Train cả 2 models và ensemble
### Do phương pháp sử dụng LLMs nên kết quả có thể lệch một vài sample nếu chạy đi chạy lại mô hình
