#!/usr/bin/env python3
"""
Startup script — Deep Research AI (Apple Silicon)

Sequential startup — everything runs in the SAME process in order:
  1. Requirements check  — abort if any package is missing.
  2. HF login + prefetch — authenticate and download all model weights.
  3. Load model          — Qwen2.5-7B Q4 into Metal (MLX), offline mode.
  4. Warmup inference    — one short generation to JIT Metal shaders + measure latency.
  5. Launch uvicorn      — IN-PROCESS so the loaded model is shared.
  6. Open browser        — once /health responds (up to 120 s wait).

Models are swapped sequentially during research (planner → unload → writer).
Memory is never shared between two large models simultaneously.

Usage:
    python scripts/start.py
"""
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
PORT = 8000

sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ── Always keep HuggingFace Hub ONLINE so models auto-download if missing ─────
os.environ["HF_HUB_OFFLINE"] = "0"   # override any stale shell export

# ── Silence ALL HuggingFace / tqdm progress bars before any HF import ─────────
# Must be set BEFORE huggingface_hub is imported for the first time.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TQDM"] = "1"  # older hf_hub versions honour this

# ── Point HF cache at local project cache/ folder ────────────────────────────
# model_manager._hf_cache() resolves this from config.yaml (storage.hf_cache).
# Setting env vars here ensures any direct huggingface_hub / mlx_lm import
# before model_manager is loaded also picks up the right location.
_LOCAL_HF_CACHE = ROOT / "cache" / "hub"
_LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME",      str(_LOCAL_HF_CACHE.parent))
os.environ.setdefault("HF_HUB_CACHE", str(_LOCAL_HF_CACHE))


def _silence_tqdm() -> None:
    """
    Silence ALL progress bars from tqdm and huggingface_hub.

    Strategy — two layers are required:
    1. Replace tqdm.tqdm (and .auto/.notebook variants) with a no-op class
       that implements every class-level method hf_hub calls:
         get_lock() / set_lock() / write()
    2. After importing huggingface_hub, also replace the reference it has
       already cached internally (hf_hub imports tqdm at module load time
       and binds the class into its own namespace — patching tqdm afterwards
       has no effect on that cached reference).
    """
    import threading
    import importlib as _il

    class _Silent:
        # class-level _lock — accessed directly as tqdm._lock by hf_hub
        _lock = threading.RLock()

        # ── instance interface ────────────────────────────────────────────
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

        # ── class-level interface called by huggingface_hub ───────────────
        @classmethod
        def get_lock(cls):               return cls._lock
        @classmethod
        def set_lock(cls, lock):         cls._lock = lock
        @classmethod
        def write(cls, s, *a, **kw):     pass

    # ── Layer 1: patch the tqdm package itself ────────────────────────────
    try:
        import tqdm as _tqdm_mod
        _tqdm_mod.tqdm = _Silent
        for _sub in ("tqdm.auto", "tqdm.notebook"):
            try:
                _m = _il.import_module(_sub)
                _m.tqdm = _Silent
            except Exception:
                pass
    except ImportError:
        pass

    # ── Layer 2: patch hf_hub's already-cached internal reference ─────────
    # hf_hub stores its own reference to tqdm in these submodules.
    # We must overwrite each one AFTER importing the module.
    _HF_TQDM_MODULES = (
        "huggingface_hub.file_download",
        "huggingface_hub._snapshot_download",
        "huggingface_hub.utils",
        "huggingface_hub.lfs",
    )
    for _mod_name in _HF_TQDM_MODULES:
        try:
            _mod = _il.import_module(_mod_name)
            # Replace every name that IS a tqdm class (could be named
            # tqdm, tqdm_auto, disabled_tqdm, etc.)
            for _attr in list(vars(_mod).keys()):
                _val = getattr(_mod, _attr, None)
                try:
                    if (
                        isinstance(_val, type)
                        and issubclass(_val, object)
                        and _attr.lower().startswith("tqdm")
                    ):
                        setattr(_mod, _attr, _Silent)
                except Exception:
                    pass
        except Exception:
            pass


_silence_tqdm()

BANNER = "=" * 60


def _banner(title: str) -> None:
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


# ── 1. Requirements check ─────────────────────────────────────────────────────

REQUIRED = {
    # Web framework
    "fastapi":                "fastapi",
    "uvicorn":                "uvicorn",
    "aiofiles":               "aiofiles",
    "multipart":              "python-multipart",
    # ML — MLX (primary runtime on Apple Silicon)
    "mlx":                    "mlx",
    "mlx_lm":                 "mlx-lm",
    # HuggingFace (prefetch + embedding)
    "huggingface_hub":        "huggingface_hub",
    "transformers":           "transformers",
    "sentence_transformers":  "sentence-transformers",
    # Vector DB
    "chromadb":               "chromadb",
    # Knowledge graph
    "networkx":               "networkx",
    # Search + scraping
    "ddgs":                   "ddgs",
    "trafilatura":            "trafilatura",
    "bs4":                    "beautifulsoup4",
    "requests":               "requests",
    # Config + utils
    "yaml":                   "pyyaml",
    "pypdf":                  "pypdf",
    "docx":                   "python-docx",
    "PIL":                    "Pillow",
    "pytesseract":            "pytesseract",
    "youtube_transcript_api": "youtube-transcript-api",
}

# Packages that are optional — warn but don't abort if missing
OPTIONAL = {"mlx", "mlx_lm", "pytesseract", "youtube_transcript_api"}


def check_requirements() -> None:
    _banner("🔍  Step 1 — Requirements check")

    cache_dir = ROOT / "cache" / "hub"
    if cache_dir.exists():
        print(f"  ✅  Local model cache  →  {cache_dir.relative_to(ROOT)}")
    else:
        print(f"  ℹ️   cache/hub not found yet — will be created during prefetch (Step 2).")

    # Use find_spec() — checks package exists on disk WITHOUT importing it.
    # importlib.import_module() triggers full module __init__ (transformers,
    # chromadb, mlx_lm etc. each take 5-15 s), making startup appear frozen.
    from importlib.util import find_spec

    missing_required = []
    for module, pip_name in REQUIRED.items():
        # find_spec returns None if not found, raises ModuleNotFoundError for
        # broken installs — both cases mean the package is missing.
        try:
            found = find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            found = False

        if found:
            print(f"  ✅  {pip_name}")
        elif module in OPTIONAL:
            print(f"  ⚠️   {pip_name}  (optional — skipping)")
        else:
            print(f"  ❌  {pip_name}  ← MISSING")
            missing_required.append(pip_name)

    if missing_required:
        print(f"\n⛔  {len(missing_required)} required package(s) missing.")
        print(f"    pip install {' '.join(missing_required)}")
        print(f"\n    Or install everything:")
        print(f"    pip install -r {ROOT / 'requirements.txt'}")
        sys.exit(1)

    print("\n✅  All required packages present.")


# ── 2. HF login + model prefetch ─────────────────────────────────────────────

def prefetch() -> None:
    _banner("🔑  Step 2 — HuggingFace login & model prefetch")
    print("  Logging in and downloading quantized model repos to disk…\n")

    from backend.model_loader import prefetch_models, _ensure_hf_login
    _ensure_hf_login()
    results = prefetch_models()

    for key, status in results.items():
        icon = "✅" if status in ("cached", "downloaded", "already on disk") else "⚠️ "
        print(f"  {icon}  [{key}] {status}")

    print("\n✅  All models cached — no re-download on next start.")


# ── HF offline toggle ────────────────────────────────────────────────────────
# HuggingFace Hub is always kept ONLINE so mlx_lm.load() can download
# missing weights automatically. If the model is already cached, mlx_lm
# will use the local cache and only fire a lightweight etag check.


def load_model() -> None:
    """
    Load the chat model into Metal/MLX memory.
    Shows a live status line: elapsed | phase | disk read speed | RAM used
    """
    _banner("🧠  Step 3 — Loading chat model into Metal (MLX 4-bit)")
    print("  Model  : Qwen2.5-7B-Instruct (MLX 4-bit)")
    print("  Source : local cache/ (project folder)")
    print("  Size   : ~4.0 GB safetensors\n")

    import psutil

    # ── psutil disk I/O on the internal disk (disk3 on this machine) ─────────
    # We track the process's own I/O bytes so we isolate mlx_lm's reads
    # from background system activity.
    _proc = psutil.Process()

    def _proc_read_bytes() -> int:
        try:
            return _proc.io_counters().read_bytes
        except Exception:
            return 0

    def _ram_used_gb() -> float:
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1e9

    def _fmt_speed(bps: int) -> str:
        if bps >= 1_000_000_000: return f"{bps/1e9:.2f} GB/s"
        if bps >= 1_000_000:     return f"{bps/1e6:.0f} MB/s"
        if bps >= 1_000:         return f"{bps/1e3:.0f} KB/s"
        return "idle"

    done  = threading.Event()
    error: list = []

    def _do_load() -> None:
        try:
            from backend.model_manager import load_chat_model
            load_chat_model()
        except Exception as e:
            error.append(e)
        finally:
            done.set()

    t = threading.Thread(target=_do_load, daemon=True)
    t.start()

    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i       = 0
    start   = time.time()
    ram0    = _ram_used_gb()

    prev_rb = _proc_read_bytes()
    prev_t  = time.time()
    disk_speed = 0
    ram_delta  = 0.0

    # Phase heuristic — inferred from disk activity + RAM growth:
    #   "resolving cache"    → disk idle, RAM flat   (symlink/JSON resolution)
    #   "reading weights"    → disk busy, RAM growing (safetensors streaming)
    #   "compiling shaders"  → disk idle, RAM stable  (Metal GPU compilation)
    phase = "resolving cache"

    def _sample() -> None:
        nonlocal prev_rb, prev_t, phase, disk_speed, ram_delta
        now_rb = _proc_read_bytes()
        now_t  = time.time()
        dt     = max(now_t - prev_t, 0.01)
        disk_speed = int((now_rb - prev_rb) / dt)
        prev_rb = now_rb
        prev_t  = now_t
        ram_delta = _ram_used_gb() - ram0

        if disk_speed > 10_000_000:            # > 10 MB/s → actively reading
            phase = "reading weights"
        elif ram_delta > 1.5 and disk_speed < 5_000_000:
            phase = "compiling Metal shaders"
        elif ram_delta < 0.3:
            phase = "resolving cache"

    last_sample = time.time()

    while not done.wait(timeout=0.1):
        now = time.time()
        if now - last_sample >= 1.0:
            _sample()
            last_sample = now

        elapsed = int(now - start)
        line = (
            f"\r  {spinner[i % len(spinner)]}  "
            f"{elapsed:>3}s │ {phase:<26} │ "
            f"disk {_fmt_speed(disk_speed):>11} │ "
            f"RAM {_ram_used_gb():.1f} GB  Δ+{ram_delta:.1f} GB"
            f"   "
        )
        print(line, end="", flush=True)
        i += 1

    print()

    if error:
        print(f"\n❌  Model load failed: {error[0]}")
        print("    Check that the MLX weights exist in cache/hub/.")
        sys.exit(1)

    elapsed   = int(time.time() - start)
    ram_delta = _ram_used_gb() - ram0
    from backend.model_manager import get_active_name
    print(f"\n✅  '{get_active_name()}' ready in {elapsed}s  "
          f"(+{ram_delta:.1f} GB RAM)")


# ── 4. Warmup inference test ──────────────────────────────────────────────────

def warmup_inference() -> None:
    """
    Run a single short generation to confirm the model is responding and
    measure first-token + total latency before the browser opens.
    This catches Metal JIT compilation stalls that happen on the very
    first forward pass after load.
    """
    _banner("⚡  Step 4 — Warmup inference test")
    print("  Prompt : 'Reply with exactly: READY'")
    print("  (measures first-token + total latency after Metal load)\n")

    from backend.model_manager import _active, generate

    if _active is None:
        print("  ⚠️  No model loaded — skipping warmup.")
        return

    prompt = "Reply with exactly the word: READY"
    t0 = time.time()
    try:
        response = generate(
            _active,
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            system_prompt="You are a test harness. Follow instructions exactly.",
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"  ✅  Response : '{response.strip()}'")
        print(f"  ✅  Latency  : {elapsed_ms} ms  ({elapsed_ms/1000:.1f} s)")
        if elapsed_ms > 10_000:
            print("  ⚠️  First-token latency > 10 s — Metal shader cache may be cold.")
            print("       Subsequent requests will be faster.")
    except Exception as e:
        print(f"  ❌  Warmup generation failed: {e}")
        print("      The server will still start; first chat request may be slow.")


# ── 5. Launch server ──────────────────────────────────────────────────────────

def launch_server() -> None:
    """
    Run uvicorn IN-PROCESS so the chat model pre-loaded by load_model()
    is already in memory when FastAPI starts handling requests.

    A background thread opens the browser once the /health endpoint
    responds, then the main thread blocks on uvicorn forever.
    """
    _banner("🚀  Step 5 — Starting server")
    print(f"  Backend  →  http://localhost:{PORT}")
    print(f"  UI       →  http://localhost:{PORT}")
    print(f"  API Docs →  http://localhost:{PORT}/docs")
    print("  Press Ctrl+C to stop\n")

    # Signal to main.py startup hook that a model is already in memory
    os.environ["MODEL_PRELOADED"] = "1"

    def _open_browser() -> None:
        url = f"http://localhost:{PORT}"
        # Give uvicorn a moment to bind before we start polling.
        # The server process starts right after this thread is spawned,
        # so a short initial sleep avoids a flood of connection-refused errors.
        time.sleep(2)
        print("⏳  Waiting for server…", end="", flush=True)
        # Poll for up to 120 s (240 × 0.5 s).  uvicorn startup after a
        # large MLX model load can take several seconds.
        for _ in range(240):
            time.sleep(0.5)
            try:
                import urllib.request
                urllib.request.urlopen(f"{url}/health", timeout=1)
                print(f"\n✅  Server up → opening {url}")
                webbrowser.open(url)
                return
            except Exception:
                print(".", end="", flush=True)
        print("\n⚠️  Server did not respond in 120 s — check logs above.")

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = "0.0.0.0",
        port    = PORT,
        workers = 1,          # must be 1 — model is in THIS process
        reload  = False,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Entry point ───────────────────────────────────────────────────────────────

def run_preflight() -> None:
    """
    Run the pre-flight check suite (checks 1-8 + 10) before the server starts.
    Hard-fails on any FAIL result; warns but continues on WARN.
    Skips the heavy model/pipeline checks (9, 11, 12) — those are done
    by load_model() + warmup_inference() below anyway.
    """
    _banner("🔬  Step 0 — Pre-flight checks")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight.py"), "--fast"],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("\n⛔  Pre-flight checks FAILED — fix the above errors before starting.")
        print("    Run  python scripts/preflight.py  for the full report.")
        sys.exit(1)
    print("\n✅  Pre-flight passed — continuing with server startup.")


if __name__ == "__main__":
    _banner("🔬  Deep Research AI — Local Engine (Apple Silicon)")
    run_preflight()        # Step 0 — pre-flight (fast checks, no model load)
    check_requirements()   # Step 1 — abort on missing packages
    prefetch()             # Step 2 — HF login + snapshot_download
    load_model()           # Step 3 — chat model into MLX/Metal
    warmup_inference()     # Step 4 — confirm model responds + measure latency
    launch_server()        # Step 5 — uvicorn + browser
