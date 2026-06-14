# Demo LoRA Adapter — Model Card

## Overview
This adapter was trained using a tiny in‑memory corpus on an AMD RX 7700 XT using ROCm 7.2.x.  
It is intended only as a demonstration of the training pipeline.

## Base Model
- microsoft/Phi-3-mini-4k-instruct

## Training Details
- 3 epochs
- BF16 compute with 4-bit quantization (QLoRA)
- LoRA (r=8, alpha=16, dropout=0.05)
- Target modules: q_proj, v_proj
- No external datasets
- Training time: ~2 seconds (on demo corpus)

## Intended Use
- Demonstration
- Testing ROCm compatibility
- Educational purposes

## Not Intended For
- Production use
- Safety‑critical applications
- High‑quality text generation