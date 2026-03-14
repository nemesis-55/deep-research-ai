"""
constants.py — Deep Research AI
================================
Single source of truth for every hard-coded value used across the backend.

Import pattern:
    from backend.constants import CHAT_MAX_TOKENS, SEARCH_PHRASES, ...

Rules:
  • NO logic here — only immutable literals and derived constants.
  • Runtime-tunable values stay in config/config.yaml (loaded via config_loader).
  • This file must never import from other backend modules to avoid circular deps.
"""
from __future__ import annotations

import datetime
from pathlib import Path

# ── Today's date (injected into system prompts to prevent year hallucinations) ─
_TODAY = datetime.date.today().strftime("%A, %d %B %Y")   # e.g. "Saturday, 14 March 2026"

# ── Project layout ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent          # deep_research_ai/
BACKEND_DIR = Path(__file__).parent                 # deep_research_ai/backend/
CONFIG_DIR  = ROOT_DIR / "config"
LOG_DIR     = ROOT_DIR / "logs"
METRICS_DIR = LOG_DIR / "metrics"
FRONTEND_DIR = ROOT_DIR / "frontend"

# ── Upload storage (local only) ───────────────────────────────────────────────
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"

# ── Server ─────────────────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# ── Chat: complexity tier token budgets ───────────────────────────────────────
# Writer model: Qwen2.5-14B MLX 4-bit ≈ 10 tok/s on M4 16 GB (upgraded from 7B @ 19.4 tok/s)
# Chat model uses same weights — token budgets apply equally.
#   trivial:        256 tok → ~26 s   ✓
#   conversational: 768 tok → ~77 s   ✓
#   technical:      1536 tok → ~154 s (still acceptable for deep-search chat)
# Model context window: 32 768 tokens — these are output-only budgets.
CHAT_MAX_TOKENS: dict[str, int] = {
    "trivial":        256,    # greetings — one clear paragraph
    "conversational": 768,    # back-and-forth — up to ~560 words
    "technical":      2048,   # web search / deep questions — long-form, ~512 s max
}

# ── Chat: system prompts per tier ─────────────────────────────────────────────
CHAT_SYSTEM_PROMPTS: dict[str, str] = {
    "trivial": (
        "You are a friendly assistant. "
        "Reply naturally and briefly — 1-2 sentences at most."
    ),
    "conversational": (
        "You are a helpful assistant. "
        "Answer clearly and concisely. "
        "Keep replies under 100 words unless more detail is genuinely needed."
    ),
    "technical": (
        f"Today is {_TODAY}. "
        "You are a knowledgeable AI research assistant with access to live web search results. "
        "When web search context is provided above the user's message, use it to give an "
        "accurate, up-to-date answer grounded in that context. "
        "Always use the CURRENT year from today's date — never substitute a different year. "
        "Cite sources inline (Title — URL). "
        "Structure your answer with clear headings, bullet points, and sections where appropriate. "
        "Use markdown: **bold** for key terms, `code` for technical values, tables for comparisons. "
        "Give accurate, comprehensive, well-structured answers as detailed as the question requires. "
        "Do not truncate — write a complete, thorough response."
    ),
}

# ── Chat: keyword sets for complexity classification ──────────────────────────
TECHNICAL_KEYWORDS: frozenset[str] = frozenset({
    "explain", "how", "why", "what", "difference", "compare", "implement",
    "algorithm", "architecture", "design", "analyse", "analyze", "evaluate",
    "research", "summarise", "summarize", "pros", "cons", "tradeoff",
    "example", "code", "function", "class", "system", "model", "theory",
    "history", "overview", "breakdown", "detail", "deep", "write", "generate",
    "list", "steps", "tutorial", "guide", "review", "paper", "concept",
})

# ── Web search intent detection ────────────────────────────────────────────────
# Any of these in a chat message triggers a live DuckDuckGo search + scrape
# before the LLM is called, giving it grounded, up-to-date context.
SEARCH_PHRASES: tuple[str, ...] = (
    "latest", "recent", "news", "today", "current", "now", "update",
    "2024", "2025", "2026", "this week", "this month", "this year",
    "just happened", "breaking", "live", "real-time", "right now",
    "dig", "find", "search", "look up", "look for", "what's happening",
    "what is happening", "price of", "stock", "share price", "market",
)

# ── Adaptive scraping thresholds (CHAT only — deep research uses config.yaml) ─
# Chat web search is intentionally limited: fast response matters more than depth.
# Deep research search limits live in config.yaml: research.max_results_per_task
#   and deep_crawl.results_per_query (both tuned by benchmark).
#
# Benchmark analysis (2026-03-15):
#   ~30% of URLs return 404 or stubs < min_useful_chars
#   Confidence gate at 0.60 is more reliable than 0.65 with noisy real-world pages
CHAT_SEARCH_MAX_RESULTS       = 10   # DDG result pool for chat  (fixed at 10)
CHAT_SCRAPE_INITIAL_URLS      = 3    # auto-mode: always scrape at least 3 pages
CHAT_SCRAPE_MAX_URLS          = 10   # forced-on mode (🌐 toggle): scrape up to 10 pages
CHAT_SCRAPE_CONFIDENCE_THRESH = 0.60  # was 0.65 — benchmark shows ~30% page fail rate
CHAT_SCRAPE_TARGET_CHARS      = 15_000
CHAT_SCRAPE_MIN_USEFUL_CHARS  = 500   # was 800 — some good pages are short (news briefs)

# Max chars injected into LLM prompt — 80k chars ≈ 20k tokens, well within 32k context window
CHAT_SCRAPE_MAX_CHARS = 80_000

# Max chars of prompt logged to terminal
PROMPT_LOG_MAX_CHARS  = 6_000

# Max chars of think-block logged to terminal
THINK_LOG_MAX_CHARS   = 4_000

# Max chars of output logged to terminal
OUTPUT_LOG_MAX_CHARS  = 6_000

# Max chars of prompt captured in the SSE think-event (UI card)
SSE_PROMPT_MAX_CHARS  = 3_000

# Max chars of output captured in the SSE think-event (UI card)
SSE_OUTPUT_MAX_CHARS  = 4_000

# ── Web scraper ────────────────────────────────────────────────────────────────
SCRAPER_TIMEOUT_S        = 15
SCRAPER_USER_AGENT       = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# ── DuckDuckGo search back-off ─────────────────────────────────────────────────
DDG_BACKOFF_DELAYS: tuple[int, ...] = (2, 4, 8)   # seconds between retries
DDG_INTER_QUERY_DELAY_S = 1.5                       # delay between fan-out queries

# ── Metrics insights: target latency per tier (seconds) ───────────────────────
# Writer/chat model: Qwen2.5-14B MLX 4-bit ≈ 10 tok/s on M4 16 GB
#   trivial:        256 tok / 10  = 26 s  → target 30 s
#   conversational: 768 tok / 10  = 77 s  → target 85 s
#   technical:      1536 tok / 10 = 154 s → target 160 s
INSIGHT_TARGET_LATENCY_S: dict[str, int] = {
    "trivial":        30,
    "conversational": 85,
    "technical":      160,
}

# Only emit a recommendation when |current - ideal| exceeds this threshold
INSIGHT_MIN_DELTA_TOKENS = 50

# Minimum number of samples before a tier is eligible for recommendations
INSIGHT_MIN_SAMPLES = 2

# ── Required packages (health check) ──────────────────────────────────────────
REQUIRED_PACKAGES: tuple[str, ...] = (
    "fastapi", "uvicorn", "transformers",
    "huggingface_hub", "sentence_transformers",
    "chromadb", "ddgs", "trafilatura",
    "bs4", "requests", "yaml", "pypdf", "docx",
    "PIL", "pytesseract", "youtube_transcript_api",
    "networkx",
)

# ── Logging ────────────────────────────────────────────────────────────────────
# Modules that should always log at DEBUG regardless of root level
DEBUG_MODULES: tuple[str, ...] = (
    "backend.model_manager",
    "backend.model_loader",
    "backend.tools.article_scraper",
    "backend.tools.web_search_engine",
    "backend.tools.web_search",
    "backend.tools.page_scraper",
    "backend.agent.research_agent",
    "backend.agent.planner_agent",
    "backend.agent.report_agent",
    "backend.pipeline.research_pipeline",
)

# Third-party modules whose noisy DEBUG output we silence to WARNING
SILENT_MODULES: tuple[str, ...] = (
    # standard HTTP clients
    "httpx", "httpcore", "urllib3", "requests",
    # HuggingFace / ML
    "transformers", "huggingface_hub",
    # ddgs internal HTTP stack (primp / hyper / rustls / h2)
    "primp", "primp.connect",
    "hyper_util", "hyper_util.client", "hyper_util.client.legacy",
    "hyper_util.client.legacy.connect", "hyper_util.client.legacy.connect.http",
    "hyper_util.client.legacy.pool",
    "rustls", "rustls.client", "rustls.client.hs", "rustls.client.tls13",
    "h2", "h2.client", "h2.codec", "h2.codec.framed_write", "h2.codec.framed_read",
    "h2.frame", "h2.frame.settings", "h2.hpack", "h2.hpack.decoder",
    "h2.proto", "h2.proto.connection", "h2.proto.settings",
    "cookie_store", "cookie_store.cookie_store",
    # scraping libraries
    "trafilatura", "trafilatura.htmlprocessing", "trafilatura.main_extractor",
    "trafilatura.readability_lxml", "trafilatura.external",
    "htmldate", "htmldate.validators",
    # file locking / chromadb / sentence-transformers
    "filelock", "filelock._unix", "filelock._windows",
    "chromadb", "chromadb.config",
    "sentence_transformers", "sentence_transformers.SentenceTransformer",
    # Suppress BertModel / SentenceTransformer load-report spam
    "sentence_transformers.models", "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "transformers.modeling_utils",       # "BertModel LOAD REPORT" source
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "torch._dynamo",
)
