/* ── App-wide state ───────────────────────────────────────────────────────── */
const API        = 'http://localhost:8000';
let currentMode  = 'chat';
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
      ? `<span class="status-dot green"></span>Ready · ${escHtml(d.role || 'chat')}`
      : `<span class="status-dot yellow"></span>Loading…`;
  } catch {
    if (nameEl)   nameEl.textContent = 'Offline';
    if (statusEl) statusEl.innerHTML = '<span class="status-dot red"></span>Backend not running';
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

/* ── Mode switch ──────────────────────────────────────────────────────────── */
function switchMode(mode) {
  currentMode = mode;
  document.getElementById('view-chat').classList.toggle('active', mode === 'chat');
  document.getElementById('view-research').classList.toggle('active', mode === 'research');
  document.getElementById('btn-chat').classList.toggle('active', mode === 'chat');
  document.getElementById('btn-research').classList.toggle('active', mode === 'research');
}

/* ── Busy state ───────────────────────────────────────────────────────────── */
const _SEND_SVG = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;
const _SRCH_SVG = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>`;

function setBusy(v) {
  isStreaming = v;
  const btn = document.getElementById('send-btn');
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
    btn.innerHTML = v ? '<div class="spinner"></div>' : _SRCH_SVG;
  }
  if (bar) bar.style.display = v ? 'flex' : 'none';
  if (nowBtn) {
    nowBtn.disabled    = false;
    nowBtn.textContent = '⚡ Report Now';
  }
}
