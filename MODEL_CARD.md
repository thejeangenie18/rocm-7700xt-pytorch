# Demo LoRA Adapter — Model Card

## Overview
This adapter was trained using a tiny in‑memory corpus on an AMD RX 7700 XT using ROCm 6.x.  
It is intended only as a demonstration of the training pipeline.

## Base Model
- EleutherAI/gpt-neo-125M

## Training Details
- 3 epochs
- BF16
- LoRA (r=8, alpha=16, dropout=0.05)
- No external datasets
- Training time: ~2 seconds

## Intended Use
- Demonstration
- Testing ROCm compatibility
- Educational purposes

## Not Intended For
- Production use
- Safety‑critical applications
- High‑quality text generation
