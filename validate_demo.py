from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER = "./demo-output"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# Configure 4-bit quantization for base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER)

prompt = "The purpose of this demo is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(out[0], skip_special_tokens=True))