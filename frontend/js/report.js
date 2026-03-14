/* ── Report: download / copy / PDF  +  Thought-process panel ─────────────── */

let _lastReportText  = '';
let _thinkStepCount  = 0;
let _researchQuery   = '';

/* ══════════════════════════════════════════════════════════════════════════
   Thought-process panel helpers  (called from research.js during streaming)
══════════════════════════════════════════════════════════════════════════ */

function resetThinkBlock() {
  _thinkStepCount = 0;
  const block = document.getElementById('think-block');
  const body  = document.getElementById('think-body');
  if (block) { block.style.display = 'none'; block.open = false; }
  if (body)  body.innerHTML = '';
  _setThinkSummary('Thinking…', true);
}

function _setThinkSummary(text, spinning) {
  const el   = document.getElementById('think-summary-text');
  const sp   = document.getElementById('think-spinner');
  const hint = document.querySelector('.think-toggle-hint');
  if (el) el.textContent = text;
  if (sp) sp.style.display = spinning ? 'inline-block' : 'none';
  if (hint) {
    hint.textContent = document.getElementById('think-block')?.open
      ? 'click to collapse' : 'click to expand';
  }
}

/** Append one step row to the thought-process panel. */
function addThinkStep(msg, kind = 'step') {
  const block = document.getElementById('think-block');
  const body  = document.getElementById('think-body');
  if (!block || !body) return;

  block.style.display = '';   // make visible on first entry
  _thinkStepCount++;

  const icons = {
    step: '⏳', done: '✅', plan: '📋', search: '🔍',
    scrape: '📄', store: '📦', error: '❌', info: 'ℹ️',
  };

  const row = document.createElement('div');
  row.className = `think-row think-${kind}`;
  row.innerHTML =
    `<span class="think-icon">${icons[kind] || '•'}</span>` +
    `<span class="think-msg">${escHtml(msg)}</span>`;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;

  _setThinkSummary(`${_thinkStepCount} steps…`, true);
}

/** Call once research finishes — stops spinner, collapses the panel. */
function finaliseThinkBlock(stepCount, sourceCount, elapsedSec) {
  _setThinkSummary(
    `${stepCount} steps · ${sourceCount} sources · ${elapsedSec}s elapsed`,
    false,
  );
  const block = document.getElementById('think-block');
  if (block) block.open = false;          // auto-collapse so report is prominent
  const hint = document.querySelector('.think-toggle-hint');
  if (hint) hint.textContent = 'click to expand';
}

/* ══════════════════════════════════════════════════════════════════════════
   AI I/O debug panel  (aio-block)
   Shows every AI interaction: role · prompt → think → response
══════════════════════════════════════════════════════════════════════════ */

let _aioCardCount = 0;

function resetAioBlock() {
  _aioCardCount = 0;
  const block = document.getElementById('aio-block');
  const body  = document.getElementById('aio-body');
  const label = document.getElementById('aio-summary-text');
  if (block) { block.style.display = 'none'; block.open = false; }
  if (body)  body.innerHTML = '';
  if (label) label.textContent = 'AI Interactions';
}

/**
 * Append one AI interaction card to the aio-block panel.
 * @param {string} role      — 'planner' | 'writer' | 'chat'
 * @param {string} prompt    — truncated prompt sent to the model
 * @param {string} output    — model response
 * @param {string} thinkText — raw <think> block content (may be empty)
 */
function addAioCard(role, prompt, output, thinkText) {
  const block = document.getElementById('aio-block');
  const body  = document.getElementById('aio-body');
  const label = document.getElementById('aio-summary-text');
  if (!block || !body) return;

  block.style.display = '';   // reveal panel on first card
  _aioCardCount++;
  if (label) label.textContent = `AI Interactions (${_aioCardCount})`;

  const roleLabel = role.charAt(0).toUpperCase() + role.slice(1);
  const roleClass = ['planner','writer','chat'].includes(role) ? role : 'chat';

  const thinkSection = thinkText
    ? `<div class="aio-section">
         <div class="aio-label">💭 Think block</div>
         <div class="aio-think-text">${escHtml(thinkText)}</div>
       </div>`
    : '';

  const card = document.createElement('div');
  card.className = 'aio-card';
  card.innerHTML = `
    <div class="aio-card-header">
      <span class="aio-role-badge ${roleClass}">${escHtml(roleLabel)}</span>
      <span>#${_aioCardCount}</span>
      <span class="aio-stats">${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="aio-section">
      <div class="aio-label">▶ Prompt</div>
      <div class="aio-text">${escHtml(prompt)}</div>
    </div>
    ${thinkSection}
    <div class="aio-section">
      <div class="aio-label">◀ Response</div>
      <div class="aio-text">${escHtml(output)}</div>
    </div>`;

  body.appendChild(card);
  body.scrollTop = body.scrollHeight;
}

/* Keep toggle-hint in sync when user manually toggles */
document.addEventListener('DOMContentLoaded', () => {
  const block = document.getElementById('think-block');
  if (!block) return;
  block.addEventListener('toggle', () => {
    const hint = document.querySelector('.think-toggle-hint');
    if (hint) hint.textContent = block.open ? 'click to collapse' : 'click to expand';
  });
});

/* ══════════════════════════════════════════════════════════════════════════
   Report actions — Copy · Download .md · Download .txt · Download PDF
══════════════════════════════════════════════════════════════════════════ */

function downloadReport(ext) {
  if (!_lastReportText) return;
  const mime = ext === 'md' ? 'text/markdown' : 'text/plain';
  const blob = new Blob([_lastReportText], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `deep-research-${_slug(_researchQuery)}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadReportPDF() {
  const el = document.getElementById('report-output');
  if (!el || !_lastReportText) return;

  // Build a print-ready HTML page, open it in a new tab, then auto-print.
  // User saves as PDF from the native print dialog (Cmd+P → Save as PDF).
  const printCSS = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Georgia, 'Times New Roman', serif;
      font-size: 14px; line-height: 1.85;
      max-width: 780px; margin: 48px auto; padding: 0 28px; color: #111;
    }
    h1 { font-size: 26px; font-weight: 800; border-bottom: 2px solid #ddd;
         padding-bottom: 10px; margin-bottom: 18px; }
    h2 { font-size: 20px; font-weight: 700; margin: 30px 0 10px; }
    h3 { font-size: 16px; font-weight: 700; margin: 22px 0 8px; color: #333; }
    p  { margin-bottom: 12px; }
    ul, ol { padding-left: 22px; margin-bottom: 12px; }
    li { margin-bottom: 4px; }
    a  { color: #1a56db; text-decoration: none; }
    code { background: #f4f4f4; border-radius: 3px; padding: 1px 5px;
           font-family: 'Courier New', monospace; font-size: 13px; }
    pre  { background: #f4f4f4; border-radius: 6px; padding: 14px;
           overflow-x: auto; font-size: 12.5px; margin-bottom: 14px; }
    pre code { background: none; padding: 0; }
    blockquote { border-left: 3px solid #bbb; margin: 0 0 12px;
                 padding-left: 14px; color: #555; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 14px; }
    th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th { background: #f0f0f0; font-weight: 700; }
    hr { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
    img { max-width: 100%; border-radius: 6px; }
    .report-footer {
      margin-top: 48px; padding-top: 12px; border-top: 1px solid #ddd;
      font-size: 11px; color: #888;
    }
    @media print {
      body { margin: 20px; }
      a::after { content: " (" attr(href) ")"; font-size: 10px; color: #666; }
    }
  `;

  const title = escHtml(_researchQuery || 'Deep Research Report');
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${title}</title>
  <style>${printCSS}</style>
</head>
<body>
  ${el.innerHTML}
  <div class="report-footer">
    Generated by Deep Research AI &nbsp;·&nbsp; ${new Date().toLocaleString()}
    &nbsp;·&nbsp; Query: ${title}
  </div>
  <script>
    window.onload = () => {
      setTimeout(() => { window.print(); }, 400);
    };
  <\/script>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url  = URL.createObjectURL(blob);
  window.open(url, '_blank');
  setTimeout(() => URL.revokeObjectURL(url), 90_000);
}

async function copyReport(evt) {
  if (!_lastReportText) return;
  try {
    await navigator.clipboard.writeText(_lastReportText);
    const btn  = evt.currentTarget;
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  } catch {
    alert('Copy failed — please select and copy the report text manually.');
  }
}

/* ── Internal utilities ──────────────────────────────────────────────────── */

function _slug(str) {
  return (str || 'report')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'report';
}
