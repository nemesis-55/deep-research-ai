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
    Download all model weights into cache/ if not already present.
    Automatically purges incomplete/corrupt partial downloads before retrying.
    Skips any repo that already has a fully complete snapshot on disk.
    """
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from backend.config_loader import get as cfg_get
    from backend.model_manager import _hf_cache

    cache   = _hf_cache()          # project_root/cache/hub  (Path)
    token   = cfg_get("huggingface.token", "") or None
    results: dict = {}

    def _repo_dir(repo: str) -> Path:
        return cache / ("models--" + repo.replace("/", "--"))

    def _has_incomplete_blobs(repo: str) -> bool:
        """True if the blobs/ dir contains any .incomplete files (stalled download)."""
        blobs = _repo_dir(repo) / "blobs"
        if not blobs.exists():
            return False
        return any(b.name.endswith(".incomplete") for b in blobs.iterdir())

    def _snapshot_exists(repo: str) -> bool:
        """
        True only when the repo has a fully complete snapshot:
          - snapshots/<hash>/config.json  must exist
          - at least one substantial weight blob (>100 KB) must be present
            and must NOT be an .incomplete file
        """
        snaps_dir = _repo_dir(repo) / "snapshots"
        if not snaps_dir.exists():
            return False
        for snap in snaps_dir.iterdir():
            if not snap.is_dir():
                continue
            if not (snap / "config.json").exists():
                continue
            # Weight files can live directly in snapshot (small models) or
            # as symlinks into blobs/ (large sharded models).
            # Resolve symlinks so we check actual file content, not just the link.
            for f in snap.iterdir():
                if f.name.endswith(".incomplete"):
                    continue
                try:
                    real = f.resolve()
                    if real.suffix in (".safetensors", ".bin", ".gguf") and real.stat().st_size > 100_000:
                        return True
                except Exception:
                    pass
            # Sharded models: check blobs/ directly for any large complete blob
            blobs_dir = _repo_dir(repo) / "blobs"
            if blobs_dir.exists():
                for b in blobs_dir.iterdir():
                    if b.name.endswith(".incomplete"):
                        continue
                    try:
                        if b.stat().st_size > 100_000:
                            return True
                    except Exception:
                        pass
        return False

    def _purge_incomplete(repo: str) -> None:
        """Remove the entire repo cache dir so snapshot_download starts fresh."""
        d = _repo_dir(repo)
        if d.exists():
            print(f"  🗑️   [{repo}] purging incomplete download ({d.name}) …")
            shutil.rmtree(d, ignore_errors=True)

    # ── MLX repos (planner + writer/chat share one repo) ─────────────────────
    seen_repos: set = set()
    for role in ("planner", "writer", "chat", "embedding"):
        cfg      = cfg_get(f"models.{role}") or {}
        repo     = cfg.get("mlx_repo") or cfg.get("hf_repo", "")
        key      = f"{role}_mlx" if "mlx_repo" in cfg else "embedding"
        if not repo or repo in seen_repos:
            if repo in seen_repos:
                results[key] = "cached"   # same weights as writer
            continue
        seen_repos.add(repo)

        # ── Detect & purge incomplete downloads before the exists-check ──────
        if _has_incomplete_blobs(repo) or (
            _repo_dir(repo).exists() and not _snapshot_exists(repo)
        ):
            _purge_incomplete(repo)

        if _snapshot_exists(repo):
            print(f"  ✅  [{key}] cached")
            results[key] = "cached"
            continue

        print(f"  ⬇️   [{key}] downloading {repo} …")
        print(f"        (large models can take several minutes on first run)")
        try:
            snapshot_download(
                repo_id         = repo,
                cache_dir       = str(cache),
                token           = token,
                ignore_patterns = ["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )
            print(f"  ✅  [{key}] downloaded")
            results[key] = "downloaded"
        except Exception as e:
            print(f"  ⚠️   [{key}] download failed: {e}")
            results[key] = f"warning: {e}"

    return results


# ── Ensure HF login (called by start.py) ─────────────────────────────────────

def _ensure_hf_login() -> None:
    from huggingface_hub import login as hf_login
    from backend.config_loader import get as cfg_get
    token = cfg_get("huggingface.token", "")
    if token:
        hf_login(token=token)
