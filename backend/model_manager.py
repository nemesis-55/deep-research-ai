"""
Model Manager — Deep Research AI (Apple Silicon / Mac Mini M4)

Runtime strategy
────────────────
Primary  → llama_cpp  (GGUF Q4_K_M — most memory-efficient on Metal)
Alt      → mlx_lm     (if llama_cpp unavailable)

16 GB memory budget
────────────────────
One model loaded at a time, max ~5.5 GB (8B Q4) or ~4.8 GB (7B Q4).
Embedding model (~100 MB) stays loaded independently and is never swapped.
clear_memory() calls mx.metal.clear_cache() + gc.collect() after every unload.

Public API
──────────
  load_planner_model()            → ModelHandle
  load_writer_model()             → ModelHandle
  load_chat_model()               → ModelHandle
  unload_model()
  swap_model(role: str)           → ModelHandle
  clear_memory()
  generate(handle, prompt, **kw)  → str
  get_active_role()               → str | None
  get_active_name()               → str | None
  get_active_runtime()            → str | None
"""
from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config_loader import get
from backend.constants import (
    PROMPT_LOG_MAX_CHARS,
    THINK_LOG_MAX_CHARS,
    OUTPUT_LOG_MAX_CHARS,
    SSE_PROMPT_MAX_CHARS,
    SSE_OUTPUT_MAX_CHARS,
)

logger = logging.getLogger(__name__)


# ── Storage helpers ───────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).parent.parent   # deep_research_ai/

def _models_dir() -> Path:
    raw = get("storage.models", "cache/models")
    p   = Path(raw) if Path(raw).is_absolute() else _project_root() / raw
    p.mkdir(parents=True, exist_ok=True)
    return p

def _hf_cache() -> Path:
    """Return the HF hub cache directory (always local, inside project cache/)."""
    import os as _os
    raw = get("storage.hf_cache", "cache/hub")
    p   = Path(raw) if Path(raw).is_absolute() else _project_root() / raw
    p.mkdir(parents=True, exist_ok=True)
    # Keep HF env vars in sync so mlx_lm.load() / snapshot_download() use this path
    _os.environ["HF_HUB_OFFLINE"] = "0"           # always online
    _os.environ["HF_HOME"]        = str(p.parent)
    _os.environ["HF_HUB_CACHE"]   = str(p)
    return p


# ── ModelHandle ───────────────────────────────────────────────────────────────

@dataclass
class ModelHandle:
    role:           str
    name:           str
    runtime:        str          # "llama_cpp" | "mlx"
    model:          Any
    tokenizer:      Any          # None for llama_cpp (internal tokenizer)
    context_length: int   = 8192
    max_new_tokens: int   = 1024
    temperature:    float = 0.7
    extra:          Dict  = field(default_factory=dict)


# ── Global state ──────────────────────────────────────────────────────────────

_lock:   threading.Lock         = threading.Lock()
_active: Optional[ModelHandle]  = None

# Queue that generate() pushes (role, think_text) into so the pipeline
# can drain and emit SSE "think" events without changing the generate() signature.
import queue as _queue
_think_queue: _queue.Queue = _queue.Queue()


# ── Memory cleanup ────────────────────────────────────────────────────────────

def clear_memory() -> None:
    """Release Metal/MLX buffers and run Python GC."""
    try:
        import mlx.core as mx   # type: ignore
        # mx.metal.clear_cache() was deprecated in MLX 0.22 — use mx.clear_cache()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()   # fallback for older mlx versions
        logger.info("[Memory] MLX cache cleared.")
    except Exception:
        pass
    gc.collect()
    logger.info("[Memory] GC collected.")


# ── Unload ────────────────────────────────────────────────────────────────────

def unload_model() -> None:
    """Dereference active model and release Metal memory."""
    global _active
    with _lock:
        if _active is None:
            return
        logger.info(f"[{_active.role.upper()}] Unloading '{_active.name}'…")
        _active = None
    clear_memory()
    logger.info("[Memory] Unload complete.")


# ── llama_cpp loader (PRIMARY) ────────────────────────────────────────────────

def _load_llama_cpp(role: str, cfg: dict) -> ModelHandle:
    from llama_cpp import Llama   # type: ignore

    gguf_path = Path(cfg["gguf_path"])

    if not gguf_path.exists():
        logger.info(f"[{role.upper()}] GGUF not on disk — downloading from HuggingFace…")
        _download_gguf(cfg)

    n_ctx = cfg["context_length"]
    logger.info(f"[{role.upper()}] Loading GGUF: {gguf_path.name}  n_ctx={n_ctx}  n_gpu_layers=-1")

    llm = Llama(
        model_path   = str(gguf_path),
        n_ctx        = n_ctx,
        n_gpu_layers = -1,       # all layers → Metal GPU
        verbose      = False,
    )
    logger.info(f"[{role.upper()}] ✅  llama_cpp model ready: {cfg['name']}")
    return ModelHandle(
        role           = role,
        name           = cfg["name"],
        runtime        = "llama_cpp",
        model          = llm,
        tokenizer      = None,
        context_length = n_ctx,
        max_new_tokens = cfg["max_new_tokens"],
        temperature    = cfg["temperature"],
    )


def _download_gguf(cfg: dict) -> None:
    """Download GGUF from HuggingFace and save it to the configured gguf_path alias.

    hf_hub_download writes to local_dir/<filename>.  We then rename to the
    short alias path so _load_llama_cpp can always find it via cfg['gguf_path'].
    """
    import shutil
    from huggingface_hub import hf_hub_download  # type: ignore

    token      = get("huggingface.token", "") or None
    dest_dir   = _models_dir()
    alias_path = Path(cfg["gguf_path"])          # e.g. …/deepseek_r1_8b_q4.gguf
    hf_name    = cfg["gguf_filename"]            # e.g. DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf

    logger.info(f"  Downloading {hf_name} from {cfg['gguf_repo']}…")
    downloaded = hf_hub_download(
        repo_id   = cfg["gguf_repo"],
        filename  = hf_name,
        local_dir = str(dest_dir),
        token     = token,
    )
    downloaded_path = Path(downloaded)

    # Rename HF filename → alias path expected by _load_llama_cpp
    if downloaded_path.resolve() != alias_path.resolve():
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_path), str(alias_path))
        logger.info(f"  Renamed {downloaded_path.name} → {alias_path.name}")

    logger.info(f"  Saved → {alias_path}")


# ── MLX loader (ALTERNATIVE) ──────────────────────────────────────────────────

def _resolve_mlx_path(repo: str) -> str:
    """
    Return the local snapshot directory for an HF repo if it exists in
    cache/hub, otherwise return the repo ID for online download.

    mlx_lm.load() does NOT accept a cache_dir kwarg — the only way to
    guarantee it reads from our local cache/ folder is to pass the full
    absolute path to the snapshot directory directly.
    """
    cache = _hf_cache()
    # HF on-disk layout: models--<org>--<name>/snapshots/<hash>/
    folder = "models--" + repo.replace("/", "--")
    snapshots_dir = cache / folder / "snapshots"
    if snapshots_dir.exists():
        # pick the first (and normally only) snapshot hash
        snaps = sorted(snapshots_dir.iterdir())
        if snaps:
            resolved = str(snaps[-1].resolve())
            logger.info(f"  → resolved local path: {resolved}")
            return resolved
    # Fallback: let mlx_lm download via HF hub (env vars already set)
    logger.warning(f"  → no local snapshot found for {repo}, will download")
    return repo


def _load_mlx(role: str, cfg: dict) -> ModelHandle:
    from mlx_lm import load as mlx_load   # type: ignore

    repo = cfg["mlx_repo"]
    path = _resolve_mlx_path(repo)

    logger.info(f"[{role.upper()}] Loading MLX: {path}")
    model, tokenizer = mlx_load(
        path,
        tokenizer_config={"trust_remote_code": True},
    )
    logger.info(f"[{role.upper()}] ✅  MLX model ready: {cfg['name']}")
    return ModelHandle(
        role           = role,
        name           = cfg["name"],
        runtime        = "mlx",
        model          = model,
        tokenizer      = tokenizer,
        context_length = cfg["context_length"],
        max_new_tokens = cfg["max_new_tokens"],
        temperature    = cfg["temperature"],
    )


# ── Core loader ───────────────────────────────────────────────────────────────

def _load_role(role: str) -> ModelHandle:
    cfg = get(f"models.{role}")
    if not cfg:
        raise ValueError(f"No model configured for role '{role}' in config.yaml")

    runtime = cfg.get("runtime", "llama_cpp")
    logger.info(f"[{role.upper()}] Requested runtime: {runtime}")

    if runtime == "llama_cpp":
        try:
            return _load_llama_cpp(role, cfg)
        except ImportError:
            logger.warning(f"[{role.upper()}] llama_cpp not installed — trying MLX…")
        except Exception as e:
            logger.warning(f"[{role.upper()}] llama_cpp failed ({e}) — trying MLX…")

    # MLX fallback
    try:
        return _load_mlx(role, cfg)
    except ImportError:
        raise RuntimeError(
            "No inference runtime available.\n"
            "Install llama-cpp-python:  pip install llama-cpp-python\n"
            "Or mlx-lm:                pip install mlx-lm"
        )


def _load(role: str) -> ModelHandle:
    """Thread-safe: auto-unload current model, then load role.
    The lock is held for the full check→unload→load sequence to
    prevent TOCTOU races when concurrent requests arrive.
    """
    global _active
    with _lock:
        if _active is not None and _active.role == role:
            logger.info(f"[{role.upper()}] Already loaded — reusing.")
            return _active
        # Unload inside the lock so no other thread can slip in between
        if _active is not None:
            logger.info(f"[{_active.role.upper()}] Unloading '{_active.name}'…")
            _active = None
        # Release Metal memory before loading the next model
    clear_memory()
    with _lock:
        handle  = _load_role(role)
        _active = handle
        return handle


# ── Public load functions ─────────────────────────────────────────────────────

def load_planner_model() -> ModelHandle:
    logger.info("=" * 55)
    logger.info("Loading PLANNER — DeepSeek-R1 8B Q4_K_M")
    logger.info("=" * 55)
    return _load("planner")

def load_writer_model() -> ModelHandle:
    logger.info("=" * 55)
    logger.info("Loading WRITER — Qwen2.5-7B Q4_K_M")
    logger.info("=" * 55)
    return _load("writer")

def load_chat_model() -> ModelHandle:
    logger.info("=" * 55)
    logger.info("Loading CHAT — Qwen2.5-7B Q4_K_M")
    logger.info("=" * 55)
    return _load("chat")

def swap_model(role: str) -> ModelHandle:
    """Unload current model then load role. Ensures only one model in memory."""
    logger.info(f"[SWAP] {get_active_role() or 'none'} → {role}")
    unload_model()
    return _load(role)


# ── Generation ────────────────────────────────────────────────────────────────

def generate(
    handle: ModelHandle,
    prompt: str,
    max_new_tokens: Optional[int]   = None,
    temperature:    Optional[float] = None,
    system_prompt:  str = "You are a helpful research assistant. Always respond in English.",
    complexity:     str = "",   # passed through from /chat for metrics logging
) -> str:
    import re
    import time as _time

    max_tok = max_new_tokens if max_new_tokens is not None else handle.max_new_tokens
    temp    = temperature    if temperature    is not None else handle.temperature
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    # ── Log full prompt to terminal ───────────────────────────────────────────
    _SEP = "─" * 60
    logger.info(f"\n{_SEP}")
    logger.info(f"[{handle.role.upper()}] ▶ PROMPT  (role={handle.role}, max={max_tok}, temp={temp})")
    logger.info(f"{_SEP}")
    for line in prompt[:PROMPT_LOG_MAX_CHARS].splitlines():
        logger.info(f"  {line}")
    if len(prompt) > PROMPT_LOG_MAX_CHARS:
        logger.info(f"  … [{len(prompt)-PROMPT_LOG_MAX_CHARS} chars truncated]")
    logger.info(_SEP)

    t0 = _time.monotonic()

    if handle.runtime == "llama_cpp":
        logger.info(f"[{handle.role.upper()}] Generating via llama_cpp (max={max_tok}, temp={temp})…")
        out = handle.model.create_chat_completion(
            messages    = messages,
            max_tokens  = max_tok,
            temperature = temp,
            top_p       = get("generation.top_p", 0.9),
        )
        response = out["choices"][0]["message"]["content"]
        _out_tokens = out.get("usage", {}).get("completion_tokens", 0)

    elif handle.runtime == "mlx":
        from mlx_lm import generate as mlx_generate              # type: ignore
        from mlx_lm.sample_utils import make_sampler             # type: ignore
        logger.info(f"[{handle.role.upper()}] Generating via mlx_lm (max={max_tok}, temp={temp})…")
        tok       = handle.tokenizer
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        top_p     = get("generation.top_p", 0.9)
        sampler   = make_sampler(temp=temp, top_p=top_p)
        response  = mlx_generate(
            handle.model, handle.tokenizer,
            prompt     = formatted,
            max_tokens = max_tok,
            sampler    = sampler,
            verbose    = False,
        )
        _out_tokens = 0
    else:
        raise ValueError(f"Unknown runtime: {handle.runtime}")

    elapsed_s = _time.monotonic() - t0

    raw_response = str(response)

    # ── Extract <think> block ─────────────────────────────────────────────────
    think_match = re.search(r"<think>(.*?)</think>", raw_response, flags=re.DOTALL)
    think_text  = ""
    if think_match:
        think_text = think_match.group(1).strip()
        logger.info(f"\n[{handle.role.upper()}] 🧠 THINK BLOCK ({len(think_text)} chars):")
        logger.info("─" * 60)
        for line in think_text[:THINK_LOG_MAX_CHARS].splitlines():
            logger.info(f"  💭 {line}")
        if len(think_text) > THINK_LOG_MAX_CHARS:
            logger.info(f"  … [{len(think_text)-THINK_LOG_MAX_CHARS} chars truncated]")
        logger.info("─" * 60)

    # Strip <think> block from final output
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    if not _out_tokens:
        _out_tokens = max(1, len(text.split()))

    # ── Log full output to terminal ───────────────────────────────────────────
    logger.info(f"[{handle.role.upper()}] ◀ OUTPUT  ({len(text)} chars / ~{_out_tokens} tok / {elapsed_s:.2f}s / {_out_tokens/elapsed_s:.1f} tok/s)")
    logger.info("─" * 60)
    for line in text[:OUTPUT_LOG_MAX_CHARS].splitlines():
        logger.info(f"  {line}")
    if len(text) > OUTPUT_LOG_MAX_CHARS:
        logger.info(f"  … [{len(text)-OUTPUT_LOG_MAX_CHARS} chars truncated]")
    logger.info("─" * 60)

    # ── Push to SSE think queue (always — so every AI call gets a card) ───────
    try:
        _think_queue.put_nowait({
            "role":   handle.role,
            "think":  think_text,
            "prompt": prompt[:SSE_PROMPT_MAX_CHARS],
            "output": text[:SSE_OUTPUT_MAX_CHARS],
        })
    except Exception:
        pass

    # ── Emit to metrics logger (non-blocking — logger has its own lock) ───────
    try:
        from backend.tools.system_metrics import get_metrics_logger
        ml = get_metrics_logger()
        if ml is not None:
            ml.log_inference(
                role          = handle.role,
                prompt_chars  = len(prompt),
                output_chars  = len(text),
                output_tokens = _out_tokens,
                elapsed_s     = elapsed_s,
                complexity    = complexity,
                model_name    = handle.name,
                runtime       = handle.runtime,
            )
    except Exception:
        pass   # metrics logging must never break generation

    return text


# ── Introspection ─────────────────────────────────────────────────────────────

def get_active_role() -> Optional[str]:
    return _active.role if _active else None

def get_active_name() -> Optional[str]:
    return _active.name if _active else None

def get_active_runtime() -> Optional[str]:
    return _active.runtime if _active else None

def drain_think_queue() -> list:
    """Drain all pending think-block items from the queue. Returns list of dicts."""
    items = []
    while True:
        try:
            items.append(_think_queue.get_nowait())
        except _queue.Empty:
            break
    return items
