from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from optimum.quanto import freeze, qfloat8, quantize

BASE = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER = "./demo-output"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# Load base model in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Apply Quanto quantization
model = quantize(model, weights=qfloat8)
freeze(model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER)

prompt = "The purpose of this demo is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(out[0], skip_special_tokens=True))