# ROCm QLoRA Demo - Fine‑Tuning on AMD RX 7700 XT

A small, reproducible demo for fine‑tuning a LoRA adapter on AMD ROCm using PyTorch.
This repository is designed for local ROCm workflows and avoids fragile cloud-dependent tooling.

Tested on a real **AMD RX 7700 XT (12GB VRAM)**.
Not included: blood, sweat, and tears after being a pipeline rat.

---

## 🚀 Overview

This demo shows how to:

- fine-tune a small transformer model with **QLoRA** on ROCm
- save a working **LoRA adapter** and tokenizer
- load the adapter for inference with `validate_demo.py`

It is intentionally minimal and stable, with a focus on reproducibility for local AMD GPU users.

---

## 📁 Repository Structure

- `training/qlora_demo_trainer.py` — ROCm-compatible QLoRA training script
- `validate_demo.py` — load the saved adapter and generate a sample response
- `requirements.txt` — Python dependencies
- `demo-output/` — produced adapter and tokenizer files

---

## ⚙️ Prerequisites

- AMD GPU with ROCm support (tested on RX 7700 XT, 12GB VRAM)
- ROCm-enabled PyTorch build
- Python 3.10–3.12
- `accelerate`, `transformers`, `peft`

> If you use a different ROCm install path, update the activation command accordingly.

---

## 🧪 Quickstart

1. Activate your ROCm environment and switch to the repo:

```bash
source ~/rocm72/bin/activate
cd /home/usr/Project/rocm-7700xt-pytorch
```

2. Run the training demo:

```bash
accelerate launch \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --num_processes=1 \
  --num_machines=1 \
  training/qlora_demo_trainer.py \
  --model_name_or_path "EleutherAI/gpt-neo-125M" \
  --output_dir "./demo-output" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --force_bf16 True
```

3. Verify the output files:

```bash
ls -la demo-output
```

Expected files:

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`

4. Validate the saved adapter:

```bash
python validate_demo.py
```

### Example `validate_demo.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "EleutherAI/gpt-neo-125M"
ADAPTER = "./demo-output"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, ADAPTER)

prompt = "The purpose of this demo is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

This script loads the base model and the trained LoRA adapter, then prints a sample generated continuation.

---

## Screenshots

### Demo Training Run
![Demo training run screenshot](Pictures/Screenshot/demo.png)
**Alt Text (DeafBlind Standard):**  
A terminal window on a dark background showing an accelerate launch command used to run a QLoRA demo trainer. The command includes flags for mixed precision bf16, dynamo backend disabled, one process, one machine, and the script `training/qlora_demo_trainer.py`. Arguments specify model `EleutherAI/gpt-neo-125M`, output directory `./demo-output`, three epochs, batch size two, gradient accumulation one, sequence length 128, learning rate 5e‑5, and `--force_bf16 True`.  
The output shows the model loading in bfloat16, a warning about deprecated `torch_dtype`, a progress bar reaching 100%, and a load report listing unexpected attention bias keys. Training metrics appear, including loss decreasing from 5.696, grad_norm around 1.2, and learning rate decay. A final summary shows runtime 2.15 seconds, samples per second 6.977, steps per second 4.186, and final loss 5.811. The run ends with saving the adapter and tokenizer to `./demo-output` and a “DONE” message.

### Demo Output Directory Listing
![Demo output directory listing screenshot](Pictures/Screenshot/demo-2.png)
**Alt Text (DeafBlind Standard):**  
A terminal window showing `ls -la demo-output` inside a ROCm virtual environment. The listing includes:  
- `adapter_config.json` (~1 KB)  
- `adapter_model.safetensors` (~1.1 MB)  
- `checkpoint-9/` directory  
- `README.md` (~5 KB)  
- `tokenizer_config.json`  
- `tokenizer.json` (~3.5 MB)  
Each entry shows permissions, owner `jg18`, group `jg18`, file sizes, timestamps, and filenames. This screenshot confirms that the LoRA adapter and tokenizer were saved correctly.

### Demo Validation Script
![Demo validation script screenshot](Pictures/Screenshot/demo-3.png)
**Alt Text (DeafBlind Standard):**  
A terminal running `python3 validate_demo.py`. The output begins with a warning about deprecated `torch_dtype`, followed by a progress bar showing weights loading to 100%. A load report for GPTNeoForCausalLM lists unexpected attention bias keys. A warning sets `pad_token_id` to EOS token 50256. A tokenizer warning appears about `clean_up_tokenization_spaces`. The script prints generated text beginning with: “The purpose of this demo is to show you how to use the new API…” and continues mid‑sentence. This screenshot confirms that the adapter loads and inference works.

## 💡 Notes

- Some `GPT-Neo` model loads may show `UNEXPECTED` key warnings for LoRA attention layers. This is expected and usually safe.
- Tokenizer cleanup warnings for GPT-Neo BPE are also harmless in this demo.
- The example training data is intentionally small, so the adapter can overfit quickly and demonstrate that the fine-tuning step worked.
- See `MODEL_CARD.md` for details about the demo adapter.

---

## 🧱 Hardware Requirements

- AMD RX 7700 XT (12GB VRAM)
- ROCm 6.x
- PyTorch ROCm build
- Python 3.10–3.12
- `accelerate`, `transformers`, `peft`

## 📌 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

- If you see `UNEXPECTED` keys when loading GPT‑Neo: this is normal for LoRA‑patched attention layers.
- If you see tokenizer cleanup warnings: harmless for GPT‑Neo BPE.
- If `accelerate` complains about config: delete `~/.cache/huggingface/accelerate/default_config.yaml`.

---

## ♿️ Accessibility & Alt‑Text Requirements

We follow a DeafBlind‑first documentation standard. Every image, screenshot, diagram, and visual asset in this repository must include accessible alt text and a long description when necessary.

What to include in alt text

- Visible text: quote or reproduce any text that appears in the image (commands, output, labels).
- Structure: describe the layout (terminal window, directory listing, table, progress bar, chart, etc.).
- Purpose / context: explain why the image is present and what the reader should notice.
- Important details: include warnings, numeric values, file names, and final statuses shown.
- No interpretation: avoid attributing intent, emotion, or uncertain meaning.

Formatting guidance

- Short vs long descriptions: keep the alt text concise but complete. If the image requires a long, multi‑sentence description, include a one‑line alt text and place the full description immediately below the image under a "Long description" heading or inside a collapsible `<details>` block.
- Markdown examples:

  - Inline image (with repository file): `![Short alt text describing image](Pictures/Screenshot/demo.png)`
  - Text‑only placeholder (no file): `![Full DeafBlind‑standard alt text goes here]()`

Alt‑text pattern (recommended)

Start your alt text with any exact visible text in quotes, then add structure and context. For example:

```
"$ accelerate launch --mixed_precision=bf16 --num_processes=1". Terminal window with dark background showing the full training log, a Loading weights progress bar at 100%, UNEXPECTED keys warning for LoRA layers, metrics table with loss and epoch summaries, and final lines: "[INFO] Saving adapter + tokenizer to ./demo-output" and "[DONE] Training complete." Context: demonstrates a successful QLoRA training run on ROCm.
```

Concrete examples

- Training screenshot alt text (short + long description below image):

  - Alt (one line): `"$ accelerate launch --mixed_precision=bf16 ...". Terminal showing a QLoRA training run and final save.`
  - Long description (below image or in `<details>`): reproduce the visible command exactly, list the important log lines (loading weights, UNEXPECTED keys note, metrics values, final save messages), and explain why this screenshot is helpful.

- Directory listing screenshot alt text:

  - `Long listing of demo-output showing files: adapter_config.json (1,024 bytes), adapter_model.safetensors (1,186,136 bytes), checkpoint-9/, README.md, tokenizer_config.json, tokenizer.json. Shows that training artifacts were written to ./demo-output.`

Contributor expectations

- When adding or updating any image in the repository (for example under README.md, docs/, examples/, or screenshots/), include accessible alt text and, if needed, a long description.
- PRs that add images without accessible alt text or a long description will be requested for changes.

Reviewer checklist

- Does the alt text reproduce visible text from the image where applicable?
- Does it describe layout and important visual cues (progress bars, warnings, sizes, filenames)?
- If the image is complex, is there a long description visible in the doc or inside a `<details>` block?

Why this matters

Accessibility is an engineering requirement for this project. Clear, literal alt text ensures the repository is useful to DeafBlind and screen‑reader users and improves overall documentation quality.

---

## Git Attributes

This repo includes a `.gitattributes` file to prevent GitHub from diffing binary model files.

---

## 📝 License

This repository is released under the **MIT License**.


```
