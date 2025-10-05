from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

app = FastAPI()

BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER_PATH = r"output_archive_name/content/sql_codellama_qlora"  # relative path inside repo

# Load everything on startup (Render CPU)
print("🚀 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)

print("🔌 Applying adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Create pipeline without device argument
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

class Query(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "message": "CodeLlama + Adapter on Render 🚀"}

@app.post("/generate")
def generate_sql(q: Query):
    result = pipe(q.text, max_new_tokens=128, do_sample=False)
    return {"input": q.text, "output": result[0]["generated_text"]}
