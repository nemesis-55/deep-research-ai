/* ══════════════════════════════════════════════════════════════════════════
   Web Crawl Activity Panel — Gemini-style live view of Phase 0
══════════════════════════════════════════════════════════════════════════ */

const _crawl = {
  queries:   0,
  urls:      0,
  stored:    0,
  sites:     0,   // Phase 1 live sites scraped
  phase0:    false,
  phase1:    false,
  collapsed: false,
};

function _crawlPhase0Start() {
  _crawl.queries   = 0;
  _crawl.urls      = 0;
  _crawl.stored    = 0;
  _crawl.sites     = 0;
  _crawl.phase0    = true;
  _crawl.collapsed = false;
  document.getElementById('crawl-queries-n').textContent  = '0';
  document.getElementById('crawl-urls-n').textContent     = '0';
  document.getElementById('crawl-stored-n').textContent   = '0';
  document.getElementById('crawl-sites-n').textContent    = '0';
  document.getElementById('crawl-phase-fill').style.width = '0%';
  document.getElementById('crawl-phase-label').textContent = 'PHASE 0 · Broad crawl';
  document.getElementById('crawl-feed').innerHTML          = '';
  document.getElementById('crawl-panel').classList.add('open');
  document.getElementById('crawl-collapse-btn').textContent = '▲ hide';
}

function _crawlPhase0End() {
  _crawl.phase0 = false;
  document.getElementById('crawl-phase-fill').style.width = '100%';
  _addCrawlRow('info', '✅',
    `Phase 0 complete — ${_crawl.stored} articles stored in vector store`, '', '');
}

function _crawlPhase1Start() {
  _crawl.phase1 = true;
  document.getElementById('crawl-phase-label').textContent = 'PHASE 1 · Deep research';
  _addCrawlRow('info', '🔬', 'Phase 1 — Deep per-task research starting…', '', '');
  document.getElementById('crawl-panel').classList.add('open');
}

function _crawlPhase1End() {
  _crawl.phase1 = false;
  document.getElementById('crawl-phase-label').textContent = 'DONE';
  _addCrawlRow('info', '✅', 'Phase 1 research complete', '', '');
}

function toggleCrawlPanel() {
  _crawl.collapsed = !_crawl.collapsed;
  const feed = document.getElementById('crawl-feed');
  const bar  = document.querySelector('.crawl-phase-bar');
  const btn  = document.getElementById('crawl-collapse-btn');
  if (_crawl.collapsed) {
    feed.style.display = 'none';
    bar.style.display  = 'none';
    btn.textContent    = '▼ show';
  } else {
    feed.style.display = '';
    bar.style.display  = '';
    btn.textContent    = '▲ hide';
  }
}

/* Progress bar: queries 20 %, URLs 50 %, stored 30 % */
function _updateCrawlProgress() {
  const pct = Math.min(100, Math.round(
    (_crawl.queries / 10)  * 20 +
    (_crawl.urls    / 100) * 50 +
    (_crawl.stored  / 30)  * 30
  ));
  document.getElementById('crawl-phase-fill').style.width = pct + '%';
}

/**
 * Append a row to the live crawl feed.
 * @param {'query'|'url'|'scrape'|'store'|'warn'|'info'} kind
 * @param {string} icon
 * @param {string} text   — main label (CSS truncates it)
 * @param {string} badge  — badge text; '' to omit
 * @param {string} cls    — badge modifier class: 'ok'|'dup'|'new'|'skip'|''
 */
function _addCrawlRow(kind, icon, text, badge, cls) {
  const feed = document.getElementById('crawl-feed');
  const row  = document.createElement('div');
  row.className = `crawl-row ${kind}`;
  row.innerHTML = `
    <span class="crawl-icon">${icon}</span>
    <span class="crawl-text">${escHtml(text)}</span>
    ${badge ? `<span class="crawl-badge ${cls}">${escHtml(badge)}</span>` : ''}`;
  feed.appendChild(row);
  // Auto-scroll only when already near the bottom
  if (feed.scrollHeight - feed.scrollTop < feed.clientHeight + 60) {
    feed.scrollTop = feed.scrollHeight;
  }
}

/**
 * Inspect a `status` SSE message and route it to the crawl panel if it is a
 * Phase-0 event.  Returns true when the message was consumed (caller should
 * then skip adding it to the step list, avoiding noise).
 *
 * @param {string} msg
 * @returns {boolean}
 */
function _routeStatusToCrawl(msg) {
  if (!msg) return false;

  /* ── Phase 0 start ────────────────────────────────────────────────── */
  if (msg.includes('Phase 0: Deep multi-query web crawl starting')) {
    _crawlPhase0Start();
    _addCrawlRow('info', '🚀', msg, '', '');
    return true;
  }

  // Only absorb further events while Phase 0 is active
  if (!_crawl.phase0) return false;

  /* ── Planner loading / query generation ──────────────────────────── */
  if (msg.includes('Loading Planner for query generation') ||
      (msg.includes('Generating') && msg.includes('queries from planner'))) {
    _addCrawlRow('info', '🧠', msg, '', '');
    return true;
  }

  /* ── "✅ N queries ready — starting fan-out search…" ─────────────── */
  if (msg.includes('queries ready')) {
    const m = msg.match(/(\d+)\s+quer(?:y|ies)\s+ready/i);
    if (m) {
      _crawl.queries = parseInt(m[1]);
      document.getElementById('crawl-queries-n').textContent = m[1];
      _updateCrawlProgress();
    }
    _addCrawlRow('query', '🔎', msg, m ? `${m[1]} queries` : '', 'new');
    return true;
  }

  /* ── "🔗 N unique URLs found — scraping top articles…" ──────────── */
  if (msg.includes('unique URLs found')) {
    const m = msg.match(/(\d+)\s+unique\s+URL/i);
    if (m) {
      _crawl.urls = parseInt(m[1]);
      document.getElementById('crawl-urls-n').textContent = m[1];
      _updateCrawlProgress();
    }
    _addCrawlRow('url', '🔗', msg, m ? `${m[1]} URLs` : '', 'ok');
    return true;
  }

  /* ── "Candidate URLs: N total → M viable after pre-filter" ──────── */
  if (msg.includes('Candidate URLs')) {
    _addCrawlRow('info', '📋', msg, '', '');
    return true;
  }

  /* ── "  [N/30] Scraping: Title…" ────────────────────────────────── */
  if (msg.match(/\[\d+\/\d+\]\s+Scraping:/)) {
    const m = msg.match(/\[(\d+)\/(\d+)\]\s+Scraping:\s*(.*)/);
    if (m) _addCrawlRow('scrape', '📄', m[3].trim(), `${m[1]}/${m[2]}`, '');
    return true;
  }

  /* ── "    ✅ Stored (N chars, cred=X) — Title" ───────────────────── */
  if (msg.includes('Stored (') && msg.includes('chars')) {
    _crawl.stored++;
    document.getElementById('crawl-stored-n').textContent = _crawl.stored;
    _updateCrawlProgress();
    const title = (msg.match(/—\s*(.+)$/) || [])[1]?.trim() ?? msg.trim();
    const cred  = (msg.match(/cred=(\d+)/) || [])[1] ?? '';
    _addCrawlRow('store', '✅', title, cred ? `cred ${cred}` : 'stored', 'ok');
    return true;
  }

  /* ── "    ⏭ Too short (N chars): url" ────────────────────────────── */
  if (msg.includes('Too short')) {
    _addCrawlRow('warn', '⏭', msg.trim(), 'short', 'skip');
    return true;
  }

  /* ── "    ⏭ Near-duplicate content: url" ─────────────────────────── */
  if (msg.includes('Near-duplicate')) {
    _addCrawlRow('warn', '⏭', msg.trim(), 'dup', 'dup');
    return true;
  }

  /* ── Scrape / store errors ────────────────────────────────────────── */
  if (msg.includes('Scrape error') || msg.includes('Store failed')) {
    _addCrawlRow('warn', '⚠️', msg.trim(), 'err', 'dup');
    return true;
  }

  /* ── "📦 Phase 0 complete / Deep crawl complete" ─────────────────── */
  if (msg.includes('Phase 0 complete') ||
      msg.toLowerCase().includes('deep crawl complete')) {
    _crawlPhase0End();
    return true;
  }

  /* ── Refine loop messages ─────────────────────────────────────────── */
  if (msg.includes('Refine') || msg.includes('refine threshold') ||
      msg.includes('Running refine pass') || msg.includes('Refine complete')) {
    _addCrawlRow('info', '🔁', msg.trim(), '', '');
    return true;
  }

  /* ── "📎 Merged N Phase-0 articles…" ─────────────────────────────── */
  if (msg.includes('Phase-0 articles')) {
    _addCrawlRow('info', '📎', msg.trim(), '', '');
    return true;
  }

  /* ── "🧩 RAG: retrieved …" ───────────────────────────────────────── */
  if (msg.includes('RAG:')) {
    _addCrawlRow('info', '🧩', msg.trim(), '', '');
    return true;
  }

  /* ── Indented sub-messages while Phase 0 is active ──────────────── */
  if (msg.startsWith('  ') || msg.startsWith('📦') || msg.startsWith('🔗')) {
    _addCrawlRow('info', '·', msg.trim(), '', '');
    return true;
  }

  /* ════════════════════════════════════════════════════════════════════
     Phase 1 — task-by-task deep research (always route to crawl panel)
     ════════════════════════════════════════════════════════════════════ */

  /* ── "✅ Writer model ready" → Phase 1 starts ────────────────────── */
  if (msg.includes('Writer model ready') || msg.includes('Writer (Qwen')) {
    _crawlPhase1Start();
    return true;
  }

  /* ── "[N/M] 🔍 Task…" — task progress header ────────────────────── */
  if (msg.match(/^\[\d+\/\d+\]/)) {
    _addCrawlRow('info', '🔬', msg.replace(/^\[\d+\/\d+\]\s*/, '').trim(), '', '');
    return true;
  }

  /* ── "🔍 Searching: …" ───────────────────────────────────────────── */
  if (msg.includes('🔍') && msg.toLowerCase().includes('search')) {
    _addCrawlRow('query', '🔎', msg.replace('🔍', '').trim(), 'searching', 'new');
    document.getElementById('crawl-panel').classList.add('open');
    return true;
  }

  /* ── "  📄 Scraping [N/M]: Title…  🔗 url" ──────────────────────── */
  if (msg.includes('📄') && msg.toLowerCase().includes('scrap')) {
    _crawl.sites++;
    document.getElementById('crawl-sites-n').textContent = _crawl.sites;
    // Extract title and URL from "📄 Scraping [N/M]: TITLE  🔗 URL"
    const urlMatch   = msg.match(/🔗\s*(\S+)/);
    const titleMatch = msg.match(/Scraping\s*(?:\[\d+\/\d+\])?\s*:\s*([^🔗]+)/i);
    const title = titleMatch ? titleMatch[1].trim() : msg.replace(/📄\s*/,'').trim();
    const url   = urlMatch   ? urlMatch[1].trim()   : '';
    const text  = url ? `${title}  —  ${url}` : title;
    _addCrawlRow('scrape', '📄', text, `#${_crawl.sites}`, 'new');
    return true;
  }

  /* ── "  🧠 Analysing: Title…" ────────────────────────────────────── */
  if (msg.includes('🧠') && msg.toLowerCase().includes('analys')) {
    const title = msg.replace(/🧠\s*Analysing:\s*/i, '').trim();
    _addCrawlRow('store', '🧠', title, 'AI', 'ok');
    return true;
  }

  /* ── "  🔄 Self-critique…" ───────────────────────────────────────── */
  if (msg.includes('🔄') || msg.toLowerCase().includes('self-critique')) {
    _addCrawlRow('info', '🔄', msg.trim(), '', '');
    return true;
  }

  /* ── "  ⚠️ Scrape failed…" ───────────────────────────────────────── */
  if (msg.includes('⚠️') || msg.toLowerCase().includes('scrape failed') ||
      msg.toLowerCase().includes('failed')) {
    _addCrawlRow('warn', '⚠️', msg.trim(), 'err', 'dup');
    return true;
  }

  /* ── "✅ Done: task — N sources" ─────────────────────────────────── */
  if (msg.includes('✅') && msg.includes('sources')) {
    _addCrawlRow('store', '✅', msg.trim(), '', 'ok');
    return true;
  }

  /* ── "📝 Writing report…" ────────────────────────────────────────── */
  if (msg.includes('Writing report') || msg.includes('📝')) {
    if (_crawl.phase1) _crawlPhase1End();
    _addCrawlRow('info', '📝', msg.trim(), '', '');
    return true;
  }

  return false;
}
