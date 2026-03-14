/* ── Markdown + code-block setup (shared by all modes) ───────────────────── */

marked.setOptions({
  breaks: true,   // single newline → <br>
  gfm:    true,   // GitHub-flavoured markdown (tables, strikethrough)
});

// Custom renderer: copy button + language label on every code block
const _renderer = new marked.Renderer();
_renderer.code = function(code, lang) {
  const escaped  = escHtml(typeof code === 'object' ? code.text : code);
  const language = (typeof code === 'object' ? code.lang : lang) || '';
  const label    = language ? `<span class="code-lang">${escHtml(language)}</span>` : '';
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
