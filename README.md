# 🔬 Deep Research AI

A fully local AI research system running on **Apple Silicon** (M4 Mac Mini, 16 GB unified memory).  
No cloud. No data leaving your machine. Everything runs on-device.

---

## Stack

| Layer | Technology |
|---|---|
| **LLM runtime** | MLX 4-bit (Apple Silicon native) |
| **Planning model** | DeepSeek-R1 8B |
| **Chat / writing model** | Qwen2.5 7B Instruct |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **Vector store** | ChromaDB |
| **Web search** | DuckDuckGo (`ddgs`) |
| **Web scraping** | trafilatura + BeautifulSoup4 |
| **Backend** | FastAPI + SSE streaming |
| **Frontend** | Vanilla HTML/CSS/JS (no build step) |
| **Storage** | Samsung T7 Shield SSD (models + uploads) |

---

## Features

- **Chat mode** — conversational AI with automatic live web search when the query needs current information
- **Deep Research mode** — multi-step pipeline: plan → search → scrape → embed → synthesise → report
- **Adaptive scraping** — confidence-scored; fetches 3–10 pages depending on query complexity
- **Conversation context** — follow-up pronouns ("it", "they", "the company") resolved from history
- **Hardware dashboard** — live CPU, RAM, GPU, VRAM, disk metrics in the sidebar
- **AI performance insights** — tok/s stats per complexity tier with adaptive budget recommendations
- **Knowledge graph** — entity/relation extraction visualised per research session
- **Pre-flight checks** — 12-check suite verifies all dependencies before server starts

---

## Setup

### 1. Clone

```zsh
git clone https://github.com/<your-username>/deep-research-ai.git
cd deep-research-ai
```

### 2. Python environment

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. HuggingFace token

```zsh
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

> Get a free token at https://huggingface.co/settings/tokens  
> Required to download gated models (DeepSeek-R1, Qwen2.5).

### 4. External drive (optional)

Mount a Samsung T7 Shield (or any drive) at `/Volumes/T7 Shield/DeepResearchAI`.  
If not present, the server falls back to local SSD automatically.

### 5. Start

```zsh
python scripts/start.py
```

This runs the 12-check pre-flight suite, then starts the server and opens the browser at `http://localhost:8000`.

---

## Pre-flight checks

```zsh
python scripts/preflight.py          # full suite (includes model load)
python scripts/preflight.py --fast   # skip model inference (checks 1–8, 10)
python scripts/preflight.py --web-only  # web search + scraper only
```

---

## Project structure

```
backend/
  main.py              # FastAPI app, chat endpoint, web search logic
  constants.py         # All tunable constants (token budgets, thresholds)
  config_loader.py     # config/config.yaml loader
  model_loader.py      # MLX / llama_cpp model loader
  model_manager.py     # generate(), prompt/think/output logging
  agent/
    planner_agent.py   # DeepSeek-R1 research planner
    research_agent.py  # per-task web search + scrape + embed
    report_agent.py    # Qwen2.5 report synthesiser
    query_planner.py   # DDG query fan-out
  pipeline/
    research_pipeline.py  # orchestrates planner → research → report
  tools/
    web_search.py      # DuckDuckGo wrapper (ddgs + fallback)
    page_scraper.py    # trafilatura + BS4 scraper
    vector_store.py    # ChromaDB store/retrieve
    knowledge_graph.py # entity/relation extraction
    system_metrics.py  # CPU/RAM/GPU/VRAM via psutil + ioreg
    source_scorer.py   # result quality scoring

config/
  config.yaml          # model names, paths, server config

frontend/
  index.html           # single-page UI
  css/main.css
  js/
    chat.js            # SSE chat, web search badge
    research.js        # SSE deep research pipeline
    metrics.js         # hardware + AI performance panel
    crawl.js           # live crawl feed panel
    kg.js              # knowledge graph panel
    report.js          # report download/copy
    app.js             # mode switching, shared state
    utils.js           # shared helpers

scripts/
  start.py             # entry point: preflight → server → browser
  preflight.py         # 12-check pre-flight suite
  analyze_metrics.py   # offline JSONL metrics analyser
```

---

## Performance (M4, 16 GB, MLX 4-bit)

| Model | Speed | Context |
|---|---|---|
| Qwen2.5-7B (chat) | ~12–14 tok/s | 2048 token budget (technical) |
| DeepSeek-R1-8B (planner) | ~10–12 tok/s | Research planning |

---

## Configuration

All tunables are in `backend/constants.py` (no restart needed for most):

| Constant | Default | Description |
|---|---|---|
| `CHAT_SCRAPE_INITIAL_URLS` | 3 | Pages always scraped |
| `CHAT_SCRAPE_MAX_URLS` | 10 | Max pages when confidence is low |
| `CHAT_SCRAPE_CONFIDENCE_THRESH` | 0.65 | Stop scraping above this score |
| `CHAT_SCRAPE_TARGET_CHARS` | 6,000 | Volume target for confidence signal |
| `CHAT_SCRAPE_MAX_CHARS` | 8,000 | Hard cap on chars injected into prompt |
| `CHAT_MAX_TOKENS.technical` | 2048 | Token budget for web-search responses |
| `CHAT_MAX_TOKENS.conversational` | 512 | Token budget for normal chat |

---

## License

MIT
