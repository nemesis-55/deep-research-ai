#!/usr/bin/env python3
"""
MiniCPM-V-2_6 MPS-only load + inference test.

Run from the project root:
    python tests/MiniCPM-V-2_6.py

What it checks:
  1. RAM available before load
  2. MPS is available — aborts immediately if not
  3. HF login
  4. Model loads directly onto MPS (processor + tokenizer + AutoModel → .to("mps"))
  5. RAM / MPS memory after load
  6. Single .chat() call with a plain text prompt
  7. Response printed to stdout

MPS only — no CPU fallback. If MPS is not available or OOM, the script exits.
"""
import os
import sys
import time
from pathlib import Path

# Disable MPS memory hard-limit — allows swap during .to("mps") transfer.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ── Allow running from project root ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN  = os.environ.get("HF_TOKEN", "")   # set in .env — never hardcode
MODEL_ID  = "openbmb/MiniCPM-V-2_6"
CACHE_DIR = "/Volumes/T7 Shield/DeepResearchAI/model_cache"
PROMPT    = "What is 2 + 2? Answer in one sentence. Reply in English only."

# ── Step 0: memory snapshot before anything ───────────────────────────────────
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"\n[RAM]  Total:     {vm.total/1e9:.1f} GB")
    print(f"[RAM]  Available: {vm.available/1e9:.1f} GB")
    print(f"[RAM]  Used:      {vm.used/1e9:.1f} GB")
except ImportError:
    print("[RAM]  psutil not installed — skipping RAM check")

import torch
print(f"\n[MPS]  Available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("❌  MPS is not available on this machine. Aborting.")
    sys.exit(1)

DEVICE = "mps"
DTYPE  = torch.float16
print(f"[MPS]  Device: {DEVICE}  dtype: {DTYPE}")

# ── Step 1: HF login ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 1 — HuggingFace login")
print('='*50)
from huggingface_hub import login as hf_login
hf_login(token=HF_TOKEN)
print("✅  HF login OK")

# ── Step 2: Load processor + tokenizer ───────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 2 — Load processor + tokenizer")
print('='*50)
from transformers import AutoProcessor, AutoTokenizer, AutoModel

t0 = time.time()
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✅  Processor + tokenizer loaded in {time.time()-t0:.1f}s")

# ── Step 3: Load model ────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 3 — Load model → MPS")
print('='*50)
print(f"  Device : {DEVICE}")
print(f"  dtype  : {DTYPE}")

t0 = time.time()
# Load to CPU first, then move to MPS in one shot.
# device_map="auto" with only 17 GB RAM offloads layers to disk (meta device)
# which breaks MiniCPM-V inference — accelerate dispatch hooks intercept
# .chat() internal calls and return garbage / raise errors.
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (set at top) lets macOS swap absorb
# the ~9 GB weight transfer without an OOM kill.
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE,          # correct kwarg (dtype= is silently ignored)
    cache_dir=CACHE_DIR,
    token=HF_TOKEN,
    # NO device_map here
)
model = model.to(DEVICE)       # single contiguous move → no meta-device layers
model.eval()
elapsed = time.time() - t0
print(f"✅  Model loaded on {DEVICE} in {elapsed:.1f}s")

# ── RAM after load ────────────────────────────────────────────────────────────
try:
    vm2 = psutil.virtual_memory()
    print(f"\n[RAM]  Available after load: {vm2.available/1e9:.1f} GB")
    print(f"[RAM]  Used after load:      {vm2.used/1e9:.1f} GB")
except Exception:
    pass

# ── Step 4: Inference ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 4 — Inference (.chat())")
print('='*50)
print(f"  Prompt: {PROMPT}\n")

# MiniCPM-V-2_6 .chat() only accepts "user" / "assistant" roles — no "system".
# Prepend the system instruction directly into the first user turn.
SYSTEM = "You are a helpful assistant. Always respond in English only."
msgs = [
    {"role": "user", "content": f"{SYSTEM}\n\n{PROMPT}"},
]

t0 = time.time()
try:
    response = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.7,
        sampling=True,
    )
except TypeError:
    # Older positional API
    response = model.chat(None, msgs, tokenizer, processor, max_new_tokens=64)

if isinstance(response, (list, tuple)):
    response = response[0]

print(f"  Response ({time.time()-t0:.1f}s): {response}")
print(f"\n{'='*50}")
print("✅  ALL STEPS PASSED — model is working correctly")
print('='*50)
