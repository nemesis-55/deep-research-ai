/* ── Shared utilities ─────────────────────────────────────────────────────── */

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;');
}

function scrollBottom(id) {
  const el = document.getElementById(id);
  if (el) el.scrollTop = el.scrollHeight;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 400) + 'px';
}

function fillResearchPrompt(text) {
  const ta = document.getElementById('research-input');
  if (!ta) return;
  ta.value = text;
  autoResize(ta);
  ta.focus();
}
