import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# Configuration & Paths
# ==========================================
MODEL_PATH = r"D:\models\DeepSeek-R1-7B"
DATASET_PATH = "test_dataset.jsonl"
OUTPUT_DIR = "./lora_test_native"

# ==========================================
# Model & Tokenizer Initialization
# ==========================================
print("[INFO] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[INFO] Loading base model in 4-bit precision...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for k-bit training to ensure gradient checkpointing stability
base_model = prepare_model_for_kbit_training(base_model)

# ==========================================
# LoRA Configuration
# ==========================================
print("[INFO] Applying LoRA adapter...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ==========================================
# Dataset Preparation
# ==========================================
print("[INFO] Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_instruction(example):
    """
    Formats the dataset into the standard prompt structure.
    Adjust the keys ('instruction', 'input', 'output') based on your actual JSONL structure.
    """
    prompt = f"User: {example.get('instruction', '')}\n{example.get('input', '')}\nAssistant: {example.get('output', '')}{tokenizer.eos_token}"
    return {"text": prompt}

formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

# ==========================================
# Training Setup & Execution
# ==========================================
print("[INFO] Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=200,          # Adjust based on your dataset size and epoch requirements
    save_steps=50,
    optim="paged_adamw_8bit", # Crucial for 8GB VRAM offloading
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none"        # Disable wandb/tensorboard for clean local runs
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,    # Restrict sequence length to prevent OOM
    tokenizer=tokenizer,
    args=training_args
)

print("[INFO] Starting model training...")
trainer.train()

print("[INFO] Saving final adapter weights...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("[INFO] Training complete.")