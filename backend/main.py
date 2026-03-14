"""
FastAPI Backend — Deep Research AI

Design pattern: Application Factory with lifespan startup hook.
  - Writer model pre-loaded at startup so the first request is instant.
  - Planner loaded on first /research call, writer reloaded after (sequential swap).
  - All AI responses streamed via Server-Sent Events (SSE).
  - Static HTML UI served directly from FastAPI (no Node.js required).

Endpoints:
  GET    /          → HTML UI
  GET    /health    → model + server status
  GET    /models    → loaded model info
  POST   /upload    → multipart file upload
  DELETE /uploads   → clear uploaded files
  POST   /chat      → streaming SSE chat
  POST   /research  → streaming SSE deep research
"""
import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional

from backend.config_loader import get
from backend.constants import (
    CHAT_MAX_TOKENS,
    CHAT_SYSTEM_PROMPTS,
    CHAT_SEARCH_MAX_RESULTS,
    CHAT_SCRAPE_MAX_CHARS,
    CHAT_SCRAPE_MIN_USEFUL_CHARS,
    CHAT_SCRAPE_INITIAL_URLS,
    CHAT_SCRAPE_MAX_URLS,
    CHAT_SCRAPE_CONFIDENCE_THRESH,
    CHAT_SCRAPE_TARGET_CHARS,
    DEBUG_MODULES,
    EXT_DRIVE,
    EXT_UPLOAD_DIR,
    FRONTEND_DIR as _FRONTEND_DIR,
    INSIGHT_MIN_DELTA_TOKENS,
    INSIGHT_MIN_SAMPLES,
    INSIGHT_TARGET_LATENCY_S,
    LOCAL_UPLOAD_DIR,
    LOG_DIR as _LOG_DIR_CONST,
    METRICS_DIR,
    REQUIRED_PACKAGES,
    SEARCH_PHRASES,
    SILENT_MODULES,
    TECHNICAL_KEYWORDS,
)

# ── Log directory (metrics JSONL files live here) ─────────────────────────────
_LOG_DIR = _LOG_DIR_CONST
# Set DEEP_RESEARCH_LOG=DEBUG in environment to enable verbose per-URL logging.
_LOG_LEVEL = getattr(logging, os.environ.get("DEEP_RESEARCH_LOG", "DEBUG").upper(), logging.DEBUG)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure our own modules always honour the chosen level
for _mod in DEBUG_MODULES:
    logging.getLogger(_mod).setLevel(logging.DEBUG)

# Keep noisy third-party libs at WARNING
for _noisy in SILENT_MODULES:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Storage paths ─────────────────────────────────────────────────────────────
FRONTEND_DIR = _FRONTEND_DIR

if EXT_DRIVE.exists():
    UPLOAD_DIR = EXT_UPLOAD_DIR
    logger.info(f"📦  Storage → external drive: {EXT_DRIVE}")
else:
    UPLOAD_DIR = LOCAL_UPLOAD_DIR
    logger.warning(
        "⚠️  T7 Shield NOT mounted — uploads falling back to LOCAL SSD. "
        "Mount the drive and restart to use external storage."
    )

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Lazy-init pipeline singleton ──────────────────────────────────────────────
_pipeline: Optional[object] = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from backend.pipeline.research_pipeline import ResearchPipeline
        _pipeline = ResearchPipeline()
    return _pipeline


# ── Health check cache (populated once at startup, not on every poll) ─────────
_health_cache: Dict = {"missing_packages": []}


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Startup → yield → shutdown."""
    logger.info(
        f"🔬 Deep Research AI — log level: "
        f"{logging.getLevelName(_LOG_LEVEL)}  "
        f"(set DEEP_RESEARCH_LOG=DEBUG for per-URL traces)"
    )

    # Package presence check — use find_spec() so we never import heavy libs
    # here. importlib.import_module() on transformers/chromadb takes 5-15 s
    # each and would block uvicorn from serving /health until all complete.
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            found = importlib.util.find_spec(pkg) is not None
        except (ModuleNotFoundError, ValueError):
            found = False
        if not found:
            missing.append(pkg)
    _health_cache["missing_packages"] = missing
    if missing:
        logger.warning(f"⚠️  Missing packages detected: {missing}")

    if not os.environ.get("MODEL_PRELOADED"):
        logger.info("ℹ️   No preloaded model — models will load lazily on first request.")
        try:
            from backend.model_loader import _ensure_hf_login
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _ensure_hf_login)
        except Exception as e:
            logger.warning(f"⚠️  HF login warning: {e}")
    else:
        logger.info("✅  Model already loaded by start.py — skipping load.")

    # ── Start background metrics logger ───────────────────────────────────────
    # Reads config for interval + retention; falls back to sensible defaults.
    _metrics_interval = get("metrics_log.interval_s", 30)
    _metrics_keep     = get("metrics_log.keep_days",   7)
    _metrics_log_dir  = Path(get("metrics_log.dir", str(_LOG_DIR / "metrics")))
    from backend.tools.system_metrics import init_metrics_logger
    _ml = init_metrics_logger(
        log_dir    = _metrics_log_dir,
        interval_s = _metrics_interval,
        keep_days  = _metrics_keep,
    )
    logger.info(
        f"📊  Metrics log → {_metrics_log_dir}  "
        f"(every {_metrics_interval}s, keep {_metrics_keep}d)"
    )

    # Pipeline is initialised lazily on first /research request — do NOT init
    # here, it imports chromadb + sentence-transformers and adds ~10 s to
    # startup, delaying the first /health response that start.py is polling for.
    logger.info("✅  Server ready — pipeline will initialise on first /research call.")

    yield  # ← server runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _ml.stop()
    logger.info("🛑  Shutting down Deep Research AI.")


# ── Application ───────────────────────────────────────────────────────────────
# NOTE: _lifespan must be defined BEFORE FastAPI() is constructed.
app = FastAPI(title="Deep Research AI", version="3.0.0", lifespan=_lifespan)

# CORS — localhost only. "allow_origins=['*'] + allow_credentials=True" is
# rejected by browsers per spec; restrict to the actual frontend origin instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",   # dev server if ever added
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ───────────────────────────────────────────────────────
# Mount frontend sub-directories so that relative paths in index.html resolve:
#   css/main.css  → /css/main.css
#   js/*.js       → /js/*.js
# Mounted AFTER middleware so the explicit API routes defined below take priority.
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js",  StaticFiles(directory=str(FRONTEND_DIR / "js")),  name="js")


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []


class ResearchRequest(BaseModel):
    query: str
    file_paths: List[str] = []


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    html_file = FRONTEND_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"message": "Deep Research AI — frontend/index.html not found"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return a minimal SVG favicon — prevents 404 noise in logs."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<text y="26" font-size="28">🔬</text>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/health")
async def health():
    """
    Returns server + model health status.
    Package check is served from _health_cache (populated at startup, not here).
    start.py polls this until model_loaded=true, then opens the browser.
    """
    import sys as _sys
    from backend.model_loader import get_loaded_model_name, get_loaded_role, get_device
    from backend.config_loader import get as cfg_get

    models_dir  = Path(cfg_get("storage.models", "/Volumes/T7 Shield/DeepResearchAI/models"))
    hf_token    = cfg_get("huggingface.token", "") or os.environ.get("HF_TOKEN", "")
    role        = get_loaded_role()
    model_ready = role is not None or bool(os.environ.get("MODEL_PRELOADED"))
    missing     = _health_cache.get("missing_packages", [])

    return {
        "status":            "ok" if (not missing and model_ready) else "degraded",
        "model_loaded":      model_ready,
        "model":             get_loaded_model_name(),
        "role":              role,
        "device":            get_device(),
        "hf_token_set":      bool(hf_token),
        "models_dir":        str(models_dir),
        "models_dir_exists": models_dir.exists(),
        "missing_packages":  missing,
        "python":            _sys.version,
        "planner":           cfg_get("models.planner.name"),
        "writer":            cfg_get("models.writer.name"),
        "chat":              cfg_get("models.chat.name"),
        "embedding":         cfg_get("models.embedding.name"),
    }


@app.get("/metrics")
async def metrics():
    """
    Live hardware metrics for the UI dashboard.
    Reads CPU/RAM via psutil, GPU/VRAM via ioreg (no sudo needed on Apple Silicon).
    Safe to poll every 3 s from the frontend.
    """
    from backend.tools.system_metrics import collect as _collect
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _collect)
        return data
    except Exception as e:
        logger.warning(f"Metrics collection failed: {e}")
        return {"error": str(e)}


@app.get("/metrics/log")
async def metrics_log(n: int = 100, type: str = ""):
    """
    Return the last N lines from today's metrics JSONL log.

    Query params:
      n    — max records to return (default 100)
      type — filter by record type: "hw" | "inference" | "" (all)

    Useful for analysis:
      GET /metrics/log?type=inference       → all generation events
      GET /metrics/log?type=hw&n=50         → last 50 hardware snapshots
    """
    from datetime import datetime as _dt
    log_dir  = _LOG_DIR / "metrics"
    log_file = log_dir / f"metrics_{_dt.now().strftime('%Y-%m-%d')}.jsonl"
    if not log_file.exists():
        return {"records": [], "file": str(log_file), "note": "No log yet for today"}

    records = []
    try:
        with open(log_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if type and obj.get("type") != type:
                        continue
                    records.append(obj)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": str(e)}

    # Return last N
    return {
        "file":    str(log_file),
        "total":   len(records),
        "records": records[-n:],
    }


@app.get("/metrics/insights")
async def metrics_insights():
    """
    Analyse the JSONL inference log and return:
      - tok/s stats per complexity tier
      - RAM / VRAM headroom
      - Optimisation recommendations (auto-tuned max_tokens budget)

    Used by the UI metrics panel and by the adaptive token budget logic.
    """
    import glob  # noqa: F401 — kept for future glob-based log discovery
    from datetime import datetime as _dt, timedelta as _td
    from statistics import mean, median

    log_dir  = _LOG_DIR / "metrics"
    # Read today + yesterday so a short session still has data
    records: list = []
    for delta in (0, 1):
        day  = (_dt.now() - _td(days=delta)).strftime("%Y-%m-%d")
        path = log_dir / f"metrics_{day}.jsonl"
        if path.exists():
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    infer   = [r for r in records if r.get("type") == "inference"]
    hw_recs = [r for r in records if r.get("type") == "hw"]

    # ── Per-complexity stats ──────────────────────────────────────────────────
    tiers: dict = {}
    for rec in infer:
        cx   = rec.get("complexity") or "unknown"
        tps  = rec.get("tok_per_s", 0)
        secs = rec.get("elapsed_s", 0)
        otok = rec.get("output_tokens", 0)
        if cx not in tiers:
            tiers[cx] = {"samples": [], "elapsed": [], "tokens": []}
        tiers[cx]["samples"].append(tps)
        tiers[cx]["elapsed"].append(secs)
        tiers[cx]["tokens"].append(otok)

    tier_stats = {}
    for cx, d in tiers.items():
        s = d["samples"]
        tier_stats[cx] = {
            "count":         len(s),
            "tok_per_s_avg": round(mean(s), 1)    if s else 0,
            "tok_per_s_med": round(median(s), 1)  if s else 0,
            "tok_per_s_min": round(min(s), 1)     if s else 0,
            "tok_per_s_max": round(max(s), 1)     if s else 0,
            "avg_elapsed_s": round(mean(d["elapsed"]), 2) if d["elapsed"] else 0,
            "avg_tokens":    round(mean(d["tokens"]), 0)  if d["tokens"]  else 0,
        }

    # ── RAM / VRAM headroom ───────────────────────────────────────────────────
    hw_ram_pct  = [r.get("ram_pct",  0) for r in hw_recs if r.get("ram_pct")]
    hw_vram_gb  = [r.get("vram_used_gb", 0) for r in hw_recs if r.get("vram_used_gb")]
    hw_gpu_pct  = [r.get("gpu_renderer_pct", 0) for r in hw_recs if r.get("gpu_renderer_pct") is not None]

    avg_ram_pct  = round(mean(hw_ram_pct), 1)  if hw_ram_pct  else None
    peak_ram_pct = round(max(hw_ram_pct),  1)  if hw_ram_pct  else None
    avg_vram_gb  = round(mean(hw_vram_gb), 2)  if hw_vram_gb  else None
    peak_vram_gb = round(max(hw_vram_gb),  2)  if hw_vram_gb  else None
    avg_gpu_pct  = round(mean(hw_gpu_pct), 1)  if hw_gpu_pct  else None

    # ── Adaptive max_tokens recommendation ────────────────────────────────────
    recommendations: list = []
    for cx, target_s in INSIGHT_TARGET_LATENCY_S.items():
        st = tier_stats.get(cx)
        if not st or st["count"] < INSIGHT_MIN_SAMPLES:
            continue
        tps = st["tok_per_s_avg"]
        if tps <= 0:
            continue
        ideal   = int(target_s * tps)
        current = _MAX_TOKENS.get(cx, 256)
        diff    = ideal - current
        if abs(diff) > INSIGHT_MIN_DELTA_TOKENS:
            direction = "↑ increase" if diff > 0 else "↓ decrease"
            recommendations.append({
                "tier":           cx,
                "current_budget": current,
                "recommended":    ideal,
                "tok_per_s":      tps,
                "target_s":       target_s,
                "direction":      direction,
                "reason": (
                    f"At {tps} tok/s, {cx} tier can produce {ideal} tokens in "
                    f"{target_s}s (current budget: {current})"
                ),
            })

    # ── Overall health summary ────────────────────────────────────────────────
    total_requests = len(infer)
    total_tokens   = sum(r.get("output_tokens", 0) for r in infer)
    all_tps        = [r.get("tok_per_s", 0) for r in infer if r.get("tok_per_s")]
    overall_tps    = round(mean(all_tps), 1) if all_tps else 0

    return {
        "total_requests":    total_requests,
        "total_tokens":      total_tokens,
        "overall_tok_per_s": overall_tps,
        "tier_stats":        tier_stats,
        "hw": {
            "avg_ram_pct":  avg_ram_pct,
            "peak_ram_pct": peak_ram_pct,
            "avg_vram_gb":  avg_vram_gb,
            "peak_vram_gb": peak_vram_gb,
            "avg_gpu_pct":  avg_gpu_pct,
        },
        "recommendations": recommendations,
        "records_analysed": len(records),
    }


@app.get("/models")
async def list_models():
    from backend.model_loader import get_loaded_model_name, get_loaded_role, get_active_runtime
    from backend.config_loader import get as cfg_get
    return {
        "active_model":   get_loaded_model_name(),
        "active_role":    get_loaded_role(),
        "active_runtime": get_active_runtime(),
        "planner":        cfg_get("models.planner.name"),
        "writer":         cfg_get("models.writer.name"),
        "chat":           cfg_get("models.chat.name"),
        "embedding":      cfg_get("models.embedding.name"),
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accept multipart file uploads and save to the uploads directory."""
    saved = []
    for file in files:
        if not file.filename:
            continue
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        size = dest.stat().st_size
        saved.append({"filename": file.filename, "path": str(dest), "size": size})
        logger.info(f"Uploaded: {file.filename} ({size:,} bytes)")
    return {"uploaded": saved}


@app.delete("/uploads")
async def clear_uploads():
    """Delete all files in the uploads directory."""
    count = 0
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Could not delete {f.name}: {e}")
    return {"cleared": count}


# ── Prompt complexity classifier ──────────────────────────────────────────────
# Buckets every chat message into one of three tiers so token budget and
# system prompt are proportional to what was actually asked.
# All phrase lists / keyword sets / token budgets live in backend/constants.py.

def _needs_web_search(message: str) -> bool:
    """Return True when the message appears to want live / current information."""
    lower = message.lower()
    return any(phrase in lower for phrase in SEARCH_PHRASES)


def _build_search_query(message: str, history: List[Dict]) -> str:
    """
    Construct a richer search query by appending key nouns from recent history.

    If the current message contains pronouns or vague references ("company",
    "it", "they") and the conversation has prior turns, extract up to 5
    meaningful words from the last user turn and prepend them so DDG has
    enough context to resolve the reference.

    Example:
        history[-1] user: "give me latest report on Reliance Industries"
        message:          "today as in 14th march news any update of company"
        → query:          "Reliance Industries today as in 14th march news any update of company"
    """
    VAGUE = {"company", "it", "they", "them", "the firm", "the stock", "that"}
    lower = message.lower()
    needs_context = any(v in lower for v in VAGUE)

    if needs_context and history:
        # Walk backwards through history to find last user message with substance
        for turn in reversed(history):
            if turn.get("role") != "user":
                continue
            prev = turn.get("content", "").strip()
            if not prev or prev.lower() == message.lower():
                continue
            # Extract first 6 meaningful words (skip stop-words / short tokens)
            stop = {"a", "an", "the", "of", "in", "on", "for", "to", "and",
                    "is", "are", "was", "were", "be", "give", "me", "any",
                    "latest", "recent", "today", "news", "update", "report"}
            key_words = [w for w in prev.split() if w.lower() not in stop][:6]
            if key_words:
                prefix = " ".join(key_words)
                query  = f"{prefix} {message}"
                logger.info(f"[Chat] 🔍 Context-enriched query: {query!r}")
                return query
            break

    return message


def _score_confidence(scraped_texts: list[str], result_snippets: list[str]) -> float:
    """
    Return a confidence score in [0.0, 1.0] representing how well the scraped
    content covers the query.

    Three signals — each contributes equally (0.33 each):

    1. Volume   — how much unique content we have vs the target chars
    2. Diversity — are snippets saying different things? (low Jaccard overlap = good)
    3. Depth    — average article depth; very short pages are low-quality

    Score ≥ CHAT_SCRAPE_CONFIDENCE_THRESH → stop scraping.
    Score <  threshold                    → fetch more pages.
    """
    # Signal 1: volume
    total_chars = sum(len(t) for t in scraped_texts)
    vol_score   = min(1.0, total_chars / CHAT_SCRAPE_TARGET_CHARS)

    # Signal 2: diversity across DDG snippets (word-set Jaccard)
    if len(result_snippets) >= 2:
        sets = [set(s.lower().split()) for s in result_snippets if s]
        # average pairwise Jaccard distance  (1 – similarity)
        overlaps: list[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                if union:
                    sim = len(sets[i] & sets[j]) / len(union)
                    overlaps.append(1.0 - sim)   # distance = diversity
        div_score = sum(overlaps) / len(overlaps) if overlaps else 0.5
    else:
        div_score = 0.3   # only 1 result → low diversity

    # Signal 3: depth — avg scraped page length ÷ 1500 chars (capped at 1.0)
    if scraped_texts:
        avg_depth = sum(len(t) for t in scraped_texts) / len(scraped_texts)
        depth_score = min(1.0, avg_depth / 1500)
    else:
        depth_score = 0.0

    score = (vol_score + div_score + depth_score) / 3.0
    logger.debug(
        f"[Chat] Confidence score: {score:.2f}  "
        f"(vol={vol_score:.2f} div={div_score:.2f} depth={depth_score:.2f})"
    )
    return score


def _chat_web_search(query: str) -> str:
    """
    Adaptive scraping: always start with CHAT_SCRAPE_INITIAL_URLS pages.
    If the confidence score is below CHAT_SCRAPE_CONFIDENCE_THRESH, keep
    fetching more URLs (up to CHAT_SCRAPE_MAX_URLS) until confidence is met
    or we run out of results.

    This means:
      - Simple, well-covered queries  → 3 pages scraped (fast, ~2-3 s)
      - Ambiguous / niche queries     → up to 10 pages scraped (thorough, ~8-12 s)
    """
    from backend.tools.web_search import search_web
    from backend.tools.page_scraper import scrape_page

    results = search_web(query, max_results=CHAT_SEARCH_MAX_RESULTS)
    if not results:
        return ""

    lines = [f"## Live web search results for: {query}\n"]
    for i, r in enumerate(results[:CHAT_SEARCH_MAX_RESULTS], 1):
        title   = r.get("title", "")
        url     = r.get("url", "")
        snippet = r.get("snippet", "").strip()
        lines.append(f"{i}. **{title}**")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet[:200]}")
        lines.append("")

    snippets       = [r.get("snippet", "") for r in results]
    scraped_texts: list[str] = []
    scraped_count  = 0
    per_page_limit = CHAT_SCRAPE_MAX_CHARS // CHAT_SCRAPE_INITIAL_URLS  # initial budget per page

    for idx, r in enumerate(results):
        url = r.get("url", "")
        if not url:
            continue

        # ── Scrape this page ──────────────────────────────────────────────────
        try:
            page = scrape_page(url)
            text = (page.get("text") or "").strip()
            if len(text) > CHAT_SCRAPE_MIN_USEFUL_CHARS:   # skip stubs/index pages
                scraped_texts.append(text)
                scraped_count += 1
                logger.debug(f"[Chat] Scraped {len(text):,} chars from {url[:60]}")
            else:
                logger.debug(f"[Chat] Skipped stub page ({len(text)} chars): {url[:60]}")
        except Exception as e:
            logger.debug(f"[Chat] Page scrape failed for {url}: {e}")

        # ── After initial pass: score confidence ──────────────────────────────
        if scraped_count == CHAT_SCRAPE_INITIAL_URLS:
            confidence = _score_confidence(scraped_texts, snippets)
            logger.info(
                f"[Chat] 📊 Confidence after {scraped_count} pages: {confidence:.2f} "
                f"(threshold={CHAT_SCRAPE_CONFIDENCE_THRESH})"
            )
            if confidence >= CHAT_SCRAPE_CONFIDENCE_THRESH:
                logger.info(f"[Chat] ✅ Confidence met — stopping at {scraped_count} pages")
                break

        # ── Hard cap ──────────────────────────────────────────────────────────
        if scraped_count >= CHAT_SCRAPE_MAX_URLS:
            logger.info(f"[Chat] 🔢 Reached max scrape limit ({CHAT_SCRAPE_MAX_URLS} pages)")
            break

    # ── Build article section with proportional budget ────────────────────────
    # Give each page a share proportional to its length so long, rich articles
    # get more of the budget and short stub pages don't waste their slot.
    if scraped_texts:
        total_len = sum(len(t) for t in scraped_texts) or 1
        for i, (r, text) in enumerate(zip(results, scraped_texts)):
            share      = len(text) / total_len
            page_chars = max(500, int(CHAT_SCRAPE_MAX_CHARS * share))
            lines.append(f"### Article {i+1}: {r.get('title', '')}")
            lines.append(f"Source: {r.get('url', '')}")
            lines.append(text[:page_chars])
            lines.append("")

    final_confidence = _score_confidence(scraped_texts, snippets) if scraped_texts else 0.0
    logger.info(
        f"[Chat] 🌐 Scraped {scraped_count} pages · "
        f"confidence={final_confidence:.2f} · "
        f"total chars={sum(len(t) for t in scraped_texts):,} · "
        f"query={query[:60]!r}"
    )
    return "\n".join(lines)


def _classify_prompt(message: str) -> str:
    """
    Return 'trivial' | 'conversational' | 'technical'.
    Web-search intent is always promoted to technical (needs 1024-token budget).

    Order matters:
      1. Web-search check FIRST — a 4-word "latest news today" must be technical,
         not trivial, because it needs live grounding and a full token budget.
      2. Trivial check — only applies when there is NO search intent.
      3. Conversational / technical split on word count + keyword presence.
    """
    stripped = message.strip()
    words    = stripped.split()
    lower    = stripped.lower()
    n        = len(words)

    # ① Web-search intent overrides everything — always needs full budget
    if _needs_web_search(message):
        return "technical"

    # ② True trivia: very short, no question mark, no search intent
    if n <= 6 and "?" not in stripped:
        return "trivial"

    # ③ Conversational vs technical
    if n <= 25 and not any(kw in lower for kw in TECHNICAL_KEYWORDS):
        return "conversational"
    return "technical"


_SYSTEM_PROMPTS = CHAT_SYSTEM_PROMPTS
_MAX_TOKENS     = CHAT_MAX_TOKENS


@app.post("/chat")
async def chat(req: ChatRequest):
    """Streaming SSE chat — with automatic live web search when needed."""

    complexity    = _classify_prompt(req.message)
    system_prompt = _SYSTEM_PROMPTS[complexity]
    max_tokens    = _MAX_TOKENS[complexity]
    logger.debug(f"[Chat] complexity={complexity} max_tokens={max_tokens} msg={req.message[:60]!r}")

    # ── Live web search (non-blocking, runs before LLM) ───────────────────────
    web_context  = ""
    web_searched = False
    if _needs_web_search(req.message):
        search_query = _build_search_query(req.message, req.history)
        logger.info(f"[Chat] 🌐 Web search triggered for: {search_query[:70]!r}")
        try:
            loop = asyncio.get_event_loop()
            web_context = await loop.run_in_executor(
                None, lambda: _chat_web_search(search_query)
            )
            web_searched = bool(web_context)
            if web_searched:
                logger.info(f"[Chat] 🌐 Web context: {len(web_context)} chars")
        except Exception as e:
            logger.warning(f"[Chat] Web search failed: {e}")

    def _build_prompt(history: List[Dict], message: str, web_ctx: str) -> str:
        lines = [f"{system_prompt}\n"]
        if web_ctx:
            lines.append(web_ctx)
            lines.append("\n---\n")
        for turn in history[-12:]:   # 12 turns ≈ 6 back-and-forth exchanges
            label = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{label}: {turn.get('content', '')}")
        lines.append(f"User: {message}\nAssistant:")
        return "\n".join(lines)

    async def event_stream():
        from backend.model_loader import generate_text

        # Signal to UI that we're searching the web
        if web_searched:
            yield _sse({"type": "web_search", "query": req.message})

        prompt = _build_prompt(req.history, req.message, web_context)
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: generate_text(
                    prompt,
                    max_new_tokens=max_tokens,
                    role="chat",
                    complexity=complexity,
                )
            )
            for i, word in enumerate(response.split(" ")):
                chunk = word if i == 0 else f" {word}"
                yield _sse({"type": "token", "content": chunk})
                await asyncio.sleep(0.008)
            yield _sse({"type": "done", "web_searched": web_searched})
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research")
async def research(req: ResearchRequest):
    """Streaming SSE deep research — Planner → Research → Report pipeline."""

    async def event_stream():
        try:
            pipeline = _get_pipeline()
            async for event in pipeline.run(
                query=req.query,
                uploaded_files=req.file_paths or [],
            ):
                yield _sse(event)
        except Exception as e:
            logger.error(f"Research error: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Entry point (prefer scripts/start.py) ─────────────────────────────────────

if __name__ == "__main__":
    host = get("server.host", "0.0.0.0")
    port = get("server.port", 8000)
    logger.info(f"🚀 Deep Research AI → http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, workers=1)
