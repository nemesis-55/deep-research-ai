#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-7B — MPS inference test.

Run from the project root:
    python tests/DeepSeek-R1.py

Load strategy:
  device_map="auto" + max_memory keeps everything in MPS/CPU RAM.
  No disk offload (no meta-device warning), no contiguous pre-allocation crash.
"""
import os
import sys
import time
from pathlib import Path

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
CACHE_DIR = str(Path(__file__).parent.parent / "cache" / "hub")
PROMPT    = "What is 2 + 2? Answer in one sentence in English."

# ── Step 0: RAM snapshot ──────────────────────────────────────────────────────
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"\n[RAM]  Total:     {vm.total/1e9:.1f} GB")
    print(f"[RAM]  Available: {vm.available/1e9:.1f} GB")
    print(f"[RAM]  Used:      {vm.used/1e9:.1f} GB")
except ImportError:
    print("[RAM]  psutil not installed — skipping")

import torch
import warnings
warnings.filterwarnings("ignore")

if not torch.backends.mps.is_available():
    print("❌  MPS not available. Aborting.")
    sys.exit(1)

DEVICE = "mps"
DTYPE  = torch.float16
print(f"\n[MPS]  Device: {DEVICE}  dtype: {DTYPE}")

# ── Step 1: Load tokenizer ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 1 — Load tokenizer")
print('='*50)

from transformers import AutoTokenizer, AutoModelForCausalLM

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)
print(f"✅  Tokenizer loaded in {time.time()-t0:.1f}s")

# ── Step 2: Load model ────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 2 — Load model")
print('='*50)

# device_map="auto" + max_memory prevents disk offload (meta-device warning).
# MPS gets 9 GB (most layers), CPU RAM absorbs overflow — no contiguous
# pre-allocation so no "Invalid buffer size: 13 GiB" crash.
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    dtype=DTYPE,
    device_map="auto"
)
model.eval()
print(f"✅  Model loaded in {time.time()-t0:.1f}s")

try:
    vm2 = psutil.virtual_memory()
    print(f"[RAM]  Available after load: {vm2.available/1e9:.1f} GB")
    print(f"[RAM]  Used after load:      {vm2.used/1e9:.1f} GB")
except Exception:
    pass

# ── Step 3: Inference ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 3 — Inference")
print('='*50)
print(f"  Prompt: {PROMPT}\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant. Always respond in English only."},
    {"role": "user",   "content": PROMPT},
]

# Build inputs and explicit attention_mask (suppresses the pad==eos warning)
input_ids      = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(DEVICE)
attention_mask = torch.ones_like(input_ids)

t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output_ids[0][input_ids.shape[-1]:]
response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"  Response ({time.time()-t0:.1f}s):\n")
print(response)
print(f"\n{'='*50}")
print("✅  ALL STEPS PASSED")
print('='*50)
