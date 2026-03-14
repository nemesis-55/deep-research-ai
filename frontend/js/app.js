/* ── App-wide state ───────────────────────────────────────────────────────── */
const API        = 'http://localhost:8000';
let currentMode  = 'normal';   // 'normal' | 'websearch' | 'research'
let isStreaming  = false;

/* ── Health check ─────────────────────────────────────────────────────────── */
async function checkHealth() {
  const nameEl   = document.getElementById('model-name');
  const statusEl = document.getElementById('model-status');
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const shortName = (d.model || '').split('/').pop() || 'Unknown';
    const ready     = d.model_loaded;
    if (nameEl)   nameEl.textContent = shortName;
    if (statusEl) statusEl.innerHTML = ready
      ? `<span class="dot green"></span>Ready · ${escHtml(d.role || 'chat')}`
      : `<span class="dot yellow"></span>Loading…`;
  } catch {
    if (nameEl)   nameEl.textContent = 'Offline';
    if (statusEl) statusEl.innerHTML = '<span class="dot red"></span>Backend not running';
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

/* ── Mode switch ──────────────────────────────────────────────────────────── */
const MODE_META = {
  normal:    { icon: '🤖', pip: '💬 Fast',       desc: 'AI only · no web search',          placeholder: 'Ask anything\u2026',                        hint: 'Enter to send · Shift+Enter for new line' },
  websearch: { icon: '🌐', pip: '🌐 Web Search',   desc: '15 relevant links · live results', placeholder: 'Ask anything \u2014 I\u2019ll search the web\u2026', hint: 'Enter to send · Shift+Enter for new line' },
  research:  { icon: '🔬', pip: '🔬 Deep Research', desc: 'Full web analysis + report',        placeholder: 'Enter research query\u2026',                hint: 'Enter to start · Shift+Enter for new line · \u26A1 Report Now to finish early' },
};

const PLACEHOLDER_META = {
  normal: {
    icon: '🤖', title: 'What can I help with?',
    desc: 'Chat with a fully local AI — no cloud, no data leaving your Mac.',
    suggestions: [
      ['Explain how unified memory works on Apple Silicon', 'Explain Apple Silicon memory'],
      ['What are the best open source LLMs in 2026?',      'Best open LLMs 2026'],
      ['Summarize the key concepts of RAG systems',        'RAG systems explained'],
      ['How does vector similarity search work?',          'Vector similarity search'],
      ['Write a Python function to parse JSON safely',     'Python JSON parser'],
      ['Explain transformer attention mechanisms',         'Transformer attention'],
    ],
  },
  websearch: {
    icon: '🌐', title: 'Search the live web',
    desc: 'I\u2019ll fetch 15 relevant sources and ground my answer in real data.',
    suggestions: [
      ['latest AI research news today',          'Latest AI news today'],
      ['recent tech breakthroughs this week',    'Recent tech breakthroughs'],
      ['current price of NVIDIA stock',          'NVIDIA stock price'],
      ['what is happening with OpenAI right now','OpenAI latest news'],
      ['best open source models released in 2026','Top models 2026'],
      ['latest developments in quantum computing','Quantum computing news'],
    ],
  },
  research: {
    icon: '🔬', title: 'Deep Research',
    desc: 'The AI will plan, search, scrape, and synthesise a full report.',
    suggestions: [
      ['Latest developments in AI chip design 2025',       'AI chip trends 2025'],
      ['OpenAI business model and revenue breakdown',       'OpenAI business model'],
      ['Apple Vision Pro market performance and reviews',   'Apple Vision Pro'],
      ['Quantum computing progress and timeline',          'Quantum computing'],
    ],
  },
};

function switchMode(mode) {
  currentMode = mode;

  // Update dropdown menu items' active state
  ['normal', 'websearch', 'research'].forEach(m => {
    const btn = document.getElementById(`btn-${m}`);
    if (btn) btn.classList.toggle('active', m === mode);
  });

  // Update trigger icon + close the menu
  const icons = { normal: '💬', websearch: '🌐', research: '🔬' };
  const iconEl  = document.getElementById('mode-drop-icon');
  const trigger = document.getElementById('mode-drop-trigger');
  const menu    = document.getElementById('mode-drop-menu');
  if (iconEl)  iconEl.textContent = icons[mode];
  if (menu)    menu.classList.remove('open');
  if (trigger) trigger.setAttribute('aria-expanded', 'false');

  const meta = MODE_META[mode];
  const ph   = PLACEHOLDER_META[mode];

  // Update textarea placeholder + hint
  const ta   = document.getElementById('research-input');
  const hint = document.getElementById('input-hint-bar');
  if (ta)   ta.placeholder   = meta.placeholder;
  if (hint) hint.textContent = meta.hint;

  // ALL modes share the unified research layout.
  const isChat      = mode === 'normal' || mode === 'websearch';
  const chatHist    = document.getElementById('chat-history');
  const reportOut   = document.getElementById('report-output');
  const innerScroll = document.getElementById('research-content-inner');
  const placeholder = document.getElementById('research-placeholder');

  // Update left-panel heading labels per mode
  const headingProgress = document.getElementById('panel-heading-progress');
  const headingSteps    = document.getElementById('panel-heading-steps');
  const headingSources  = document.getElementById('panel-heading-sources');
  if (headingProgress) headingProgress.textContent = isChat ? 'Session'  : 'Progress';
  if (headingSteps)    headingSteps.textContent    = 'Steps';
  if (headingSources)  headingSources.textContent  = 'Sources';

  // Only update placeholder when no active content is showing.
  // When switching modes, always reset content and show the new placeholder.
  const chatHasContent   = chatHist && chatHist.children.length > 0;
  const reportHasContent = reportOut && reportOut.style.display !== 'none';

  // Always reset to placeholder when switching — start fresh per mode
  if (chatHist)    { chatHist.innerHTML = ''; chatHist.style.display = 'none'; }
  if (innerScroll) innerScroll.style.display = 'flex';
  if (reportOut)   reportOut.style.display   = 'none';
  if (placeholder) placeholder.style.display = '';

  // Clear left-panel state for the new mode
  const stepList   = document.getElementById('step-list');
  const sourceList = document.getElementById('sources-list');
  const progBar    = document.getElementById('progress-bar');
  const progLabel  = document.getElementById('progress-label');
  const srcCount   = document.getElementById('source-count');
  if (stepList)   stepList.innerHTML   = '';
  if (sourceList) sourceList.innerHTML = '';
  if (srcCount)   srcCount.textContent = '';
  if (progBar)    progBar.style.width  = '0%';
  if (progLabel)  progLabel.textContent = 'Idle';

  // Reset think/aio panels
  if (typeof resetThinkBlock === 'function') resetThinkBlock();
  if (typeof resetAioBlock   === 'function') resetAioBlock();

  // Reset chat history array (defined in research.js)
  if (typeof _chatHistory !== 'undefined') _chatHistory.length = 0;

  const icon  = document.getElementById('placeholder-icon');
  const title = document.getElementById('placeholder-title');
  const pdesc = document.getElementById('placeholder-desc');
  const grid  = document.getElementById('placeholder-suggestions');
  if (icon)  icon.textContent  = ph.icon;
  if (title) title.textContent = ph.title;
  if (pdesc) pdesc.textContent = ph.desc;
  if (grid) {
    grid.innerHTML = ph.suggestions.map(([query, label]) =>
      `<button class="suggestion" onclick="fillResearchPrompt(${JSON.stringify(query)})">${escHtml(label)}</button>`
    ).join('');
  }

  // Research status bar + show-now button only for deep research
  const rsb    = document.getElementById('research-status-bar');
  const nowBtn = document.getElementById('show-now-btn');
  if (rsb && !isStreaming) rsb.style.display = 'none';
  if (nowBtn) nowBtn.style.display = mode === 'research' ? '' : 'none';

  // KG panel only relevant for deep research
  const kgPanel = document.getElementById('kg-panel');
  if (kgPanel) {
    if (mode === 'research') {
      kgPanel.classList.remove('hidden');
    } else {
      kgPanel.classList.add('hidden');
    }
  }
}

/* ── Mode dropdown open/close ─────────────────────────────────────────────── */
function toggleModeDropdown(e) {
  e.stopPropagation();
  const menu    = document.getElementById('mode-drop-menu');
  const trigger = document.getElementById('mode-drop-trigger');
  const isOpen  = menu.classList.toggle('open');
  trigger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}

// Close dropdown when clicking anywhere outside
document.addEventListener('click', () => {
  const menu    = document.getElementById('mode-drop-menu');
  const trigger = document.getElementById('mode-drop-trigger');
  if (menu)    menu.classList.remove('open');
  if (trigger) trigger.setAttribute('aria-expanded', 'false');
});

/* ── Busy state ───────────────────────────────────────────────────────────── */
const _SEND_SVG = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;

/* Single send button now shared by all modes */
function setBusy(v) {
  isStreaming = v;
  const btn = document.getElementById('research-btn');
  if (!btn) return;
  btn.disabled  = v;
  btn.innerHTML = v ? '<div class="spinner"></div>' : _SEND_SVG;
}

function setResearchBusy(v) {
  const btn    = document.getElementById('research-btn');
  const bar    = document.getElementById('research-status-bar');
  const nowBtn = document.getElementById('show-now-btn');
  if (btn) {
    btn.disabled  = v;
    btn.innerHTML = v ? '<div class="spinner"></div>' : _SEND_SVG;
  }
  /* Status bar only shows for deep research mode */
  if (bar) bar.style.display = (v && currentMode === 'research') ? 'flex' : 'none';
  if (nowBtn) {
    nowBtn.disabled    = false;
    nowBtn.textContent = '⚡ Report Now';
  }
}

/* ── Init: switchMode('normal') is called at the bottom of research.js
        (the last script loaded) so all helpers are guaranteed to exist. ── */
