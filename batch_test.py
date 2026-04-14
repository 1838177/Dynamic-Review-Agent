import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# File paths configuration
model_path = r"D:\models\DeepSeek-R1-7B"
lora_path = "./lora_test_native"
output_file = "evaluation_results.json"

# Prepare 20 test cases combining 5 subjects and different review days
notes_pool = [
    "【高等数学】泰勒公式：将一些复杂函数近似为一个多项式。核心思想是用在某一点的信息描述其附近的信息。公式包含 f(x) 的各阶导数。",
    "【计算机网络】TCP三次握手：客户端发送SYN，服务端回复SYN+ACK，客户端再回复ACK。目的是确认双方的接收和发送能力都正常。",
    "【法学基础】无过错责任原则：无论行为人有无过错，法律规定应当承担民事责任的，行为人应当承担民事责任。常见于高空坠物、饲养动物致人损害等。",
    "【医学解剖】心脏结构：分为左心房、左心室、右心房、右心室。左心室负责将富含氧气的血液泵入主动脉，流向全身，因此心肌最厚。",
    "【经济学】沉没成本：是指已经发生且不可收回的成本。在进行理性决策时，不应该考虑沉没成本，而只看未来的收益和边际成本。"
]

# Generate test dataset
test_dataset = []
days = [1, 3, 7, 1] * 5 
for i in range(20):
    test_dataset.append({
        "id": i + 1,
        "day": days[i],
        "note": notes_pool[i % 5]
    })

# Load base model safely with 4-bit quantization to prevent OOM
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)

# Mount LoRA adapter
print("Mounting LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# Response generation function
def generate_response(day, note, use_lora=True):
    instruction = f"你是一个智能复习助手。请根据用户的笔记和当前的复习阶段，生成复习内容。今天是用户的【第{day}天】复习。"
    prompt = f"User: {instruction}\n{note}\nAssistant: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Disable LoRA adapter dynamically if evaluating base model
    context_manager = model.disable_adapter() if not use_lora else torch.no_grad()
    
    with context_manager:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=400,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode and truncate prompt
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Clear VRAM cache
    del inputs, outputs
    torch.cuda.empty_cache()
    
    return response

# Execute batch inference
results = {"base_model": [], "lora_model": []}

print("\n[Phase 1/2] Running inference with Base Model (LoRA disabled)...")
for data in tqdm(test_dataset, desc="Base Model"):
    ans = generate_response(data["day"], data["note"], use_lora=False)
    results["base_model"].append({"day": data["day"], "text": ans})

print("\n[Phase 2/2] Running inference with Fine-Tuned Model (LoRA enabled)...")
for data in tqdm(test_dataset, desc="LoRA Model"):
    ans = generate_response(data["day"], data["note"], use_lora=True)
    results["lora_model"].append({"day": data["day"], "text": ans})

# Save results to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nBatch inference completed successfully. Results saved to {output_file}")