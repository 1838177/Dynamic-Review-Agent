import re
import json

# ==========================================
# Configuration
# ==========================================
RESULTS_FILE = "evaluation_results.json"

# ==========================================
# Evaluation Logic
# ==========================================
def evaluate_instruction_following(day, model_output):
    """
    Evaluates a single model output for formatting compliance.
    Returns: (has_think_tag, has_title, total_score)
    """
    # 1. Check for complete <think> blocks (allowing cross-line matching)
    has_think = 1 if re.search(r"<think>.*?</think>", model_output, flags=re.DOTALL) else 0
    
    # 2. Check for dynamic header structure, e.g., "【Day 1" or "【Day 7"
    expected_title_pattern = rf"【Day {day}.*?】"
    has_title = 1 if re.search(expected_title_pattern, model_output) else 0
    
    # 3. Calculate binary average score for this instance
    total_score = (has_think + has_title) / 2.0
    
    return has_think, has_title, total_score

def run_evaluation(model_name, outputs):
    """
    Aggregates metrics for a specific model group and prints the summary.
    """
    total_think = 0
    total_title = 0
    total_score = 0
    count = len(outputs)
    
    if count == 0:
        print(f"[{model_name}] No data found to evaluate.")
        return

    for item in outputs:
        think_s, title_s, score_s = evaluate_instruction_following(item["day"], item["text"])
        total_think += think_s
        total_title += title_s
        total_score += score_s
        
    print(f"[{model_name}] Evaluation Results (Sample Size: {count}):")
    print(f" |- <think> Tag Generation Rate : {total_think / count * 100:.1f}%")
    print(f" |- [Day X] Header Accuracy     : {total_title / count * 100:.1f}%")
    print(f" |- Overall Compliance Score    : {total_score / count * 100:.1f}%\n")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] Loading results from {RESULTS_FILE}...\n")
    
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {RESULTS_FILE}. Please run batch_test.py first.")
        exit(1)

    base_outputs = data.get("base_model", [])
    lora_outputs = data.get("lora_model", [])

    print("====================================================================")
    print(" Dimension 1: Instruction Intent & Format Compliance")
    print("====================================================================\n")

    run_evaluation("Base Model (DeepSeek-R1-7B)", base_outputs)
    run_evaluation("Fine-Tuned Model (LoRA_Native)", lora_outputs)