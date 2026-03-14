/* ── Chat ────────────────────────────────────────────────────────────────── */

// ── Configure marked.js once ──────────────────────────────────────────────
marked.setOptions({
  breaks: true,       // single newline → <br>
  gfm:    true,       // GitHub-flavoured markdown (tables, strikethrough)
});

// Custom renderer: adds copy button + language label to every code block
const _renderer = new marked.Renderer();
_renderer.code = function(code, lang) {
  const escaped = escHtml(typeof code === 'object' ? code.text : code);
  const language = (typeof code === 'object' ? code.lang : lang) || '';
  const label = language ? `<span class="code-lang">${escHtml(language)}</span>` : '';
  return `<div class="code-wrap">
    <div class="code-toolbar">${label}<button class="code-copy-btn" onclick="copyCodeBlock(this)" title="Copy code">📋 Copy</button></div>
    <pre><code class="language-${escHtml(language)}">${escaped}</code></pre>
  </div>`;
};
marked.use({ renderer: _renderer });

/** Copy a code block's content to clipboard */
function copyCodeBlock(btn) {
  const pre  = btn.closest('.code-wrap').querySelector('pre code');
  const text = pre ? pre.innerText : '';
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  }).catch(() => alert('Copy failed'));
}

let chatHistory = [];

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
}

async function handleSend() {
  const input = document.getElementById('prompt-input');
  const text  = input.value.trim();
  if (!text || isStreaming) return;

  document.getElementById('chat-empty').style.display = 'none';
  input.value = '';
  input.style.height = 'auto';

  appendMsg('user', text);
  chatHistory.push({ role: 'user', content: text });

  const aiEl      = appendMsg('ai', '');
  const contentEl = aiEl.querySelector('.msg-content');
  const metaEl    = aiEl.querySelector('.msg-meta');
  const actionsEl = aiEl.querySelector('.msg-actions');
  setBusy(true);
  let fullText    = '';
  let webSearched = false;

  try {
    const res = await fetch(`${API}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, history: chatHistory.slice(-12) }),
    });

    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let buf      = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          switch (ev.type) {
            case 'web_search':
              webSearched = true;
              contentEl.innerHTML =
                `<span class="chat-searching">🌐 Searching the web…</span>`;
              break;
            case 'token':
              fullText += ev.content;
              contentEl.innerHTML = marked.parse(fullText);
              scrollBottom('chat-messages');
              break;
            case 'done':
              if (webSearched && metaEl) {
                metaEl.innerHTML = '🌐 Answer grounded with live web search';
                metaEl.style.display = 'block';
              }
              // Show copy + PDF buttons after response completes
              if (actionsEl && fullText) {
                actionsEl.style.display = 'flex';
              }
              break;
            case 'error':
              contentEl.innerHTML =
                `<span style="color:var(--red)">Error: ${escHtml(ev.message)}</span>`;
              break;
          }
        } catch { /* malformed SSE line — skip */ }
      }
    }
  } catch (e) {
    contentEl.innerHTML =
      `<span style="color:var(--red)">Connection error: ${escHtml(e.message)}</span>`;
  }

  chatHistory.push({ role: 'assistant', content: fullText });

  // Store text on element for copy/pdf actions
  if (actionsEl) actionsEl.dataset.text = fullText;

  setBusy(false);
}

function appendMsg(role, text) {
  const wrap = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="msg-body">
      <div class="msg-role">${role === 'user' ? 'You' : 'Deep Research AI'}</div>
      <div class="msg-content">${
        text ? marked.parse(text) : '<span class="pulse">▋</span>'
      }</div>
      <div class="msg-meta" style="display:none"></div>
      ${role === 'ai' ? `<div class="msg-actions" style="display:none" data-text="">
        <button class="msg-action-btn" onclick="copyMsgText(this)" title="Copy response">📋 Copy</button>
        <button class="msg-action-btn" onclick="downloadMsgPDF(this)" title="Save as PDF">⬇ PDF</button>
      </div>` : ''}
    </div>`;
  wrap.appendChild(div);
  scrollBottom('chat-messages');
  return div;
}

/** Copy a single chat message to clipboard */
function copyMsgText(btn) {
  const text = btn.closest('.msg-actions').dataset.text || '';
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  }).catch(() => alert('Copy failed'));
}

/** Open a print-ready PDF for a single chat message */
function downloadMsgPDF(btn) {
  const actions  = btn.closest('.msg-actions');
  const text     = actions.dataset.text || '';
  const bodyEl   = btn.closest('.msg-body');
  const contentEl = bodyEl ? bodyEl.querySelector('.msg-content') : null;
  if (!contentEl || !text) return;

  const printCSS = `
    *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
    body { font-family: Georgia, serif; font-size:14px; line-height:1.85;
           max-width:780px; margin:48px auto; padding:0 28px; color:#111; }
    h1 { font-size:24px; font-weight:800; border-bottom:2px solid #ddd;
         padding-bottom:10px; margin-bottom:16px; }
    h2 { font-size:19px; font-weight:700; margin:28px 0 10px; }
    h3 { font-size:15px; font-weight:700; margin:20px 0 6px; color:#333; }
    p  { margin-bottom:12px; }
    ul,ol { padding-left:22px; margin-bottom:12px; }
    li { margin-bottom:4px; }
    a  { color:#1a56db; }
    code { background:#f4f4f4; border-radius:3px; padding:1px 5px;
           font-family:'Courier New',monospace; font-size:12.5px; }
    pre  { background:#f4f4f4; border-radius:6px; padding:14px;
           overflow-x:auto; font-size:12px; margin-bottom:14px; }
    pre code { background:none; padding:0; }
    blockquote { border-left:3px solid #bbb; padding-left:14px; color:#555; margin-bottom:12px; }
    table { border-collapse:collapse; width:100%; font-size:13px; margin-bottom:14px; }
    th,td { border:1px solid #ccc; padding:8px 12px; text-align:left; }
    th { background:#f0f0f0; font-weight:700; }
    .code-toolbar { display:none; }
    .footer { margin-top:40px; padding-top:10px; border-top:1px solid #ddd;
              font-size:11px; color:#888; }
    @media print { body { margin:20px; } a::after { content:" ("attr(href)")"; font-size:10px; color:#666; } }
  `;

  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Deep Research AI — Response</title><style>${printCSS}</style></head>
    <body>${contentEl.innerHTML}
    <div class="footer">Generated by Deep Research AI &nbsp;·&nbsp; ${new Date().toLocaleString()}</div>
    <script>window.onload=()=>setTimeout(()=>window.print(),400);<\/script>
    </body></html>`;

  const blob = new Blob([html], { type:'text/html' });
  const url  = URL.createObjectURL(blob);
  window.open(url, '_blank');
  setTimeout(() => URL.revokeObjectURL(url), 90_000);
}
