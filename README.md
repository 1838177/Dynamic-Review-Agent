# Spaced-Repetition-LoRA: Dynamic Review Agent

An LLM agent fine-tuned on DeepSeek-R1-7B for spaced repetition scenarios, dynamically restructuring and integrating knowledge points based on the Ebbinghaus forgetting curve.

## **1. The Vision**

While the Ebbinghaus Forgetting Curve is widely acknowledged as the "golden rule" for maximizing learning efficiency, its actual implementation is often hindered by high execution costs. 

This project builds an intelligent learning hub that bridges **time-based scheduling** with **dynamic prompt generation**. By fine-tuning a generative LLM via LoRA, the model learns to provide hierarchical review materials tailored to specific memory stages—reducing forgetting while drastically improving review efficiency.

## **2. Problem vs. Solution**

In practice, executing spaced repetition fails for two main reasons: **High Cognitive Load** (constantly switching between different textbooks, PPTs, and fragmented notes) and **Scheduling Failure** (forgetting not only what you learned, but when you are supposed to review it).

By treating the LLM as an automated learning hub, we transition from a manual, static process to an automated, dynamic one:

| Traditional Review | Dynamic LLM Review |
| :--- | :--- |
| **Scattered Materials:** Jumping between multiple apps, PDFs, and notebooks. | **Cross-Disciplinary Integration:** The LLM aggregates all relevant daily knowledge into one cohesive summary. |
| **Static Content:** Reading the exact same text block on Day 1 and Day 14. | **Dynamic Difficulty:** Detailed explanations on Day 1; targeted quizzes and fill-in-the-blanks on Day 7 to fight memory decay. |
| **Manual Scheduling:** Relying on complex spreadsheets or calendar apps. | **Automated Prompting:** Simply feed class notes to the model, and it takes over the timeline. |

## **3. Dataset Composition**

High-quality fine-tuning requires high-quality data. The model was trained on a custom, carefully curated dataset of 600 entries to ensure it can handle everything from complex code to abstract philosophy:

```text
- [201 Entries] Computer Science & Code
  Data structures, algorithms, computer networks, Java features, and MySQL principles. Teaches the model to handle English technical terms, code blocks, and logical deduction.

- [201 Entries] Frontier Engineering & Interdisciplinary
  Smart connected vehicle technologies (radar sensors, path planning algorithms) and AI applications in oncology. Trains the model to parse dense, cutting-edge academic terminology.

- [99 Entries] Humanities & Philosophy
  Albert Camus's Absurdism, The Myth of Sisyphus, Marxism, and literary movements. Enables the model to distill long, abstract philosophical arguments.

- [99 Entries] Daily Tech & General Science
  Wireless audio tech (LDAC encoding, Active Noise Cancellation principles). Ensures the model remains grounded in explaining everyday scientific concepts clearly.
```

## **4. FAQ**

### **Q1: Why use an LLM for summarizing instead of just reading my original notes?**
**A:** Traditional notes are "dead" text. No matter how many times you review them, you are looking at the exact same words, making it hard to test true retention. This project leverages the LLM not just to organize scattered information, but to alter the output based on the memory curve. On Day 1, it provides deep, comprehensive explanations. By Day 7, when you need active recall, it throws core concepts and fill-in-the-blank questions at you. Static notes simply cannot do this.

### **Q2: Why choose DeepSeek-R1-7B for LoRA fine-tuning?**
**A:** The core task here is note distillation and integration, not open-ended complex reasoning. DeepSeek-R1-7B offers exceptional Chinese comprehension, logical structuring, and an unbeatable cost-to-performance ratio. A 7B parameter model drastically lowers the training threshold (runnable on consumer GPUs) and saves massive compute costs for future local deployments, mobile ports, or API calls.

### **Q3: Why only open-source the weights and dataset without a UI/Frontend?**
**A:** The original intent of this project (which serves as the core of my university coursework) is to explore the underlying algorithmic feasibility of combining LLMs with memory curves. The essence of open source is providing infrastructure. The `lora_test_native` weights can be mounted onto any deployed base model, and the 600-entry dataset is highly valuable for similar educational research. I highly encourage and welcome the open-source community to fork this and build application layers (UI, WeChat Mini Programs, etc.) via PRs!

### **Q4: What are the hardware requirements to run or train this?**
**A:** Very low. The LoRA fine-tuning process was successfully executed on an 8GB VRAM laptop (RTX 4070) in a native Windows environment using the `paged_adamw_8bit` memory optimizer. For inference, the 4-bit quantized model only requires about 5.5GB of VRAM, making it accessible to the vast majority of modern consumer PCs.

## **5. Installation & Quick Start**

### **Step 1: Environment Setup**
Install the necessary dependencies. This project requires `torch`, `transformers`, `peft`, and `bitsandbytes` for 4-bit quantization.

```bash
pip install torch transformers peft datasets trl bitsandbytes
```

### **Step 2: Inference (Using the Fine-Tuned Weights)**
Once you have the base DeepSeek-R1-7B model and have downloaded the `lora_test_native` weights from the Releases page, you can use the following snippet to run the dynamic review agent:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "path/to/DeepSeek-R1-7B"
adapter_id = "./lora_test_native"

# Load model with 4-bit quantization to save VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True, 
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, adapter_id)

# Example: Day 7 Review (Testing Phase)
note = "TCP Three-way Handshake: SYN, SYN+ACK, ACK."
prompt = f"User: You are a review assistant. Today is [Day 7] of the user's review.\n{note}\nAssistant: "

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### **Step 3: Reproducing Benchmarks**
To replicate the evaluation results shown in this report, execute the following commands in your terminal:

```bash
python batch_test.py
python eval_dim1.py
```

## **6. Repository Structure**

- **`train_native.py`**: The core fine-tuning script. It handles 4-bit quantization loading, LoRA configuration, and the SFT (Supervised Fine-Tuning) process optimized for 8GB VRAM.
- **`batch_test.py`**: A validation script that runs inference on 20 distinct test cases across both the base model and the LoRA model to generate side-by-side comparison data.
- **`eval_dim1.py`**: The automated scoring engine. It uses regex and NLP matching to quantify how well the model follows the `<think>` tag and `[Day X]` header requirements.
- **`test_dataset.jsonl`**: The curated training dataset containing 601 high-quality entries across computer science, engineering, philosophy, and general tech.
- **`sft_evaluation_report.txt`**: The final quantitative benchmark report detailing the performance improvements in strategy divergence and format compliance.

---

## **💡 A Note from the Developer**

I'm currently a sophomore navigating through my own coursework and exams, so my engineering skills and free time are still pretty limited. I originally built this simply because I was tired of drowning in scattered notes and constantly forgetting what I needed to review.

While the code is still a rough prototype, I genuinely believe the idea of combining LLMs with spaced repetition can actually help us survive finals week. If you're a developer who finds this concept interesting, I would be absolutely thrilled if you could take this and run with it. Whether it's slapping a frontend UI on it, cleaning up my code, or tweaking the prompts—please feel free to fork it. Any PRs, advice, or contributions are hugely appreciated!