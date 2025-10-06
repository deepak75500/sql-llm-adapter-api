import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from flask import Flask, request, jsonify

# -----------------------------
# 1. Model Settings
# -----------------------------
BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER_ID = "tamilanda/my-sql-model"

# Optional: Optimize CPU performance
torch.set_num_threads(4)

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔹 Loading base model (CPU)... This may take a few minutes on first start")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("🔹 Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.to("cpu")
model.eval()

# -----------------------------
# 2. Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "✅ CodeLlama SQL API is running!"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query.strip():
        return jsonify({"error": "Query is empty"}), 400

    prompt = f"<s>[INST] {user_query} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"query": user_query, "response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
