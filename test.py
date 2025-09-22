import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# === Paths ===
model_path = "/Users/rsiddiq2/out-llama1b-hours"

#test_path  = "data/hours_test.jsonl"
test_path  = "data/hours_test.jsonl"
output_path = "test_results_llama1b_HIL.json"

# === Custom stopping criterion ===
class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_str):
        super().__init__()
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

# === Load model ===
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # or torch.bfloat16 if supported
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
model.to(device)

def normalize_json(txt: str):
    """Parse JSON safely and return canonical string."""
    try:
        obj = json.loads(txt)
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return txt.strip()

results = []
y_true, y_pred = [], []

# === Run test set ===
with open(test_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Running test set"):
        rec = json.loads(line)
        user = rec["conversations"][0]["value"]
        gold = normalize_json(rec["conversations"][1]["value"])

        # Wrap in llama3 template
        prompt = f"<|user|>\n{user}\n<|assistant|>"
        inputs = tok(prompt, return_tensors="pt").to(device)

        stop_criteria = StoppingCriteriaList([StopOnString(tok, "}]")])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                stopping_criteria=stop_criteria
            )

        decoded = tok.decode(out[0], skip_special_tokens=True)

        # Keep only assistant response
        if "<|assistant|>" in decoded:
            decoded = decoded.split("<|assistant|>")[-1].strip()

        # Trim to last full JSON array
        if "]" in decoded:
            decoded = decoded[: decoded.rfind("]") + 1]

        # Parse/normalize
        try:
            json.loads(decoded)
            pred = normalize_json(decoded)
        except Exception:
            pred = decoded.strip()

        results.append({"input": user, "gold": gold, "prediction": pred})
        y_true.append(gold)
        y_pred.append(pred)

# === Save raw results ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(results)} results to {output_path}")

def structurally_equal(a, b):
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return False

# === Compute metrics ===
matches = [1 if structurally_equal(g, p) else 0 for g, p in zip(y_true, y_pred)]

acc = sum(matches) / len(matches)
precision = precision_score(matches, [1]*len(matches), zero_division=0)
recall    = recall_score(matches, [1]*len(matches), zero_division=0)
f1        = f1_score(matches, [1]*len(matches), zero_division=0)

# Show mismatches
for inp, g, p, m in zip([r["input"] for r in results], y_true, y_pred, matches):
    if m == 0:
        print("\nMismatch example:")
        print("Input:", inp)
        print("Gold :", g)
        print("Pred :", p)

print("\n***** Test Metrics *****")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
