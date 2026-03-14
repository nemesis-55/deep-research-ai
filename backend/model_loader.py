"""
Model Loader — compatibility shim for Deep Research AI

All heavy logic now lives in backend/model_manager.py.
This module keeps the existing public API intact so main.py,
agents, and the pipeline need no call-site changes.

Public API (unchanged):
    get_model()             → (tokenizer, model)  [chat model]
    get_planner_model()     → (tokenizer, model)
    get_writer_model()      → (tokenizer, model)
    generate_text(prompt, max_new_tokens, role, temperature) → str
    free_memory()
    get_loaded_model_name() → str
    get_loaded_role()       → str | None
    get_device()            → str
    prefetch_models()       → dict
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Re-export memory helpers directly ─────────────────────────────────────────
from backend.model_manager import (          # noqa: E402
    clear_memory,
    get_active_name,
    get_active_role,
    get_active_runtime,
    load_chat_model,
    load_planner_model,
    load_writer_model,
    swap_model,
    unload_model,
    generate,
)

# ── Shim helpers ──────────────────────────────────────────────────────────────

def _handle_for_role(role: str):
    """Return (or load) the ModelHandle for the given role."""
    from backend.model_manager import _active, _load
    if _active is not None and _active.role == role:
        return _active
    return _load(role)


# ── Legacy tuple API — agents call get_model() / get_planner_model() etc. ─────
# The returned (tokenizer, model) tuple is only used by the old DummyMiniCPM
# path; the real generate path now goes through generate_text() below.

def get_model() -> Tuple:
    handle = load_chat_model()
    return handle.tokenizer, handle

def get_planner_model() -> Tuple:
    handle = load_planner_model()
    return handle.tokenizer, handle

def get_writer_model() -> Tuple:
    handle = load_writer_model()
    return handle.tokenizer, handle


# ── Main generation entry-point ───────────────────────────────────────────────

def generate_text(
    prompt:         str,
    max_new_tokens: Optional[int]   = None,
    role:           str             = "chat",
    temperature:    Optional[float] = None,
    complexity:     str             = "",
) -> str:
    """
    Load the model for *role* (or reuse if already active), generate, return text.
    role values: "planner" | "writer" | "chat"
    complexity: "trivial" | "conversational" | "technical" — forwarded to metrics logger.
    """
    handle = _handle_for_role(role)
    return generate(handle, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, complexity=complexity)


# ── Memory / status helpers ───────────────────────────────────────────────────

def free_memory() -> None:
    """Unload active model and release device memory. Called by pipeline swap."""
    unload_model()

def get_loaded_model_name() -> str:
    return get_active_name() or "none"

def get_loaded_role() -> Optional[str]:
    return get_active_role()

def get_device() -> str:
    runtime = get_active_runtime()
    if runtime == "mlx":
        return "mlx (Apple Silicon GPU)"
    if runtime == "llama_cpp":
        return "metal (llama_cpp)"
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "unknown"


# ── Pre-flight prefetch (called by start.py before server starts) ─────────────

def prefetch_models() -> dict:
    """
    Download all model weights to disk without loading into GPU memory.
    MLX repos → snapshot_download.
    GGUF files → hf_hub_download if not already on disk.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download, hf_hub_download
    from backend.config_loader import get as cfg_get
    from backend.model_manager import _hf_cache, _models_dir

    # Ensure HF progress bars are off regardless of import order.
    # hf_hub reads this flag lazily so setting it here (before any
    # snapshot_download call) is always effective.
    import os as _os
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Also patch hf_hub's internal tqdm reference — the module caches
    # it at import time so the env var alone is not always enough.
    import threading as _threading, importlib as _il
    class _Silent:
        _lock = _threading.RLock()
        def __init__(self, *a, **kw):        pass
        def __iter__(self):                  return iter([])
        def __len__(self):                   return 0
        def __enter__(self):                 return self
        def __exit__(self, *a):              pass
        def update(self, *a, **kw):          pass
        def close(self, *a, **kw):           pass
        def set_description(self, *a, **kw): pass
        def set_postfix(self, *a, **kw):     pass
        def refresh(self, *a, **kw):         pass
        def reset(self, *a, **kw):           pass
        @classmethod
        def get_lock(cls):                   return cls._lock
        @classmethod
        def set_lock(cls, lock):             cls._lock = lock
        @classmethod
        def write(cls, s, *a, **kw):         pass

    for _mod_name in (
        "huggingface_hub.file_download",
        "huggingface_hub._snapshot_download",
        "huggingface_hub.utils",
        "huggingface_hub.lfs",
    ):
        try:
            _mod = _il.import_module(_mod_name)
            for _attr in list(vars(_mod).keys()):
                _val = getattr(_mod, _attr, None)
                try:
                    if isinstance(_val, type) and _attr.lower().startswith("tqdm"):
                        setattr(_mod, _attr, _Silent)
                except Exception:
                    pass
        except Exception:
            pass

    token   = cfg_get("huggingface.token", "") or None
    cache   = _hf_cache()
    results: dict = {}

    roles = ["planner", "writer", "chat"]
    for role in roles:
        cfg  = cfg_get(f"models.{role}")
        name = cfg.get("name", role)

        # ── MLX repo ──────────────────────────────────────────────────────
        mlx_repo = cfg.get("mlx_repo", "")
        if mlx_repo:
            try:
                logger.info(f"[prefetch] [{role}] MLX repo: {mlx_repo}")
                snapshot_download(
                    repo_id         = mlx_repo,
                    cache_dir       = cache,
                    token           = token,
                    ignore_patterns = ["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
                )
                results[f"{role}_mlx"] = "cached"
                logger.info(f"[prefetch] [{role}] MLX ✅")
            except Exception as e:
                results[f"{role}_mlx"] = f"warning: {e}"
                logger.warning(f"[prefetch] [{role}] MLX warning: {e}")

        # ── GGUF file (fallback) ──────────────────────────────────────────
        gguf_path = Path(cfg.get("gguf_path", ""))
        if gguf_path.exists():
            results[f"{role}_gguf"] = "already on disk"
        elif cfg.get("gguf_repo") and cfg.get("gguf_filename"):
            try:
                logger.info(f"[prefetch] [{role}] GGUF: {cfg['gguf_filename']}")
                hf_hub_download(
                    repo_id   = cfg["gguf_repo"],
                    filename  = cfg["gguf_filename"],
                    local_dir = str(_models_dir()),
                    token     = token,
                )
                results[f"{role}_gguf"] = "downloaded"
                logger.info(f"[prefetch] [{role}] GGUF ✅")
            except Exception as e:
                results[f"{role}_gguf"] = f"warning: {e}"
                logger.warning(f"[prefetch] [{role}] GGUF warning: {e}")

    # ── Embedding (HF snapshot, no GPU load) ─────────────────────────────
    emb_repo = cfg_get("models.embedding.hf_repo", "nomic-ai/nomic-embed-text-v1")
    try:
        snapshot_download(repo_id=emb_repo, cache_dir=cache, token=token)
        results["embedding"] = "cached"
    except Exception as e:
        results["embedding"] = f"warning: {e}"

    return results


# ── Ensure HF login (called by start.py) ─────────────────────────────────────

def _ensure_hf_login() -> None:
    from huggingface_hub import login as hf_login
    from backend.config_loader import get as cfg_get
    token = cfg_get("huggingface.token", "")
    if token:
        hf_login(token=token)
