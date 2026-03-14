/* ── Deep Research — SSE pipeline + step list + source cards ─────────────── */

function handleResearchKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startResearch(); }
}

/** Called when user clicks ⚡ Now — fires a POST to interrupt the pipeline */
async function showResultNow() {
  const btn = document.getElementById('show-now-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Stopping…'; }
  try {
    await fetch(`${API}/research/show-now`, { method: 'POST' });
  } catch (e) {
    console.warn('show-now failed:', e);
  }
}

async function startResearch() {
  const input = document.getElementById('research-input');
  const query = input.value.trim();
  if (!query || isStreaming) return;

  // Route Normal and Web Search modes to the chat handler instead
  if (currentMode === 'normal' || currentMode === 'websearch') {
    return _handleChatMode(query);
  }

  return _handleDeepResearch(query);
}

/* ── Chat history state ───────────────────────────────────────────────────── */
let _chatHistory = [];   // [{role:'user'|'assistant', content:'...'}]

/* ── Null-safe getElementById helpers ──────────────────────────────────────── */
function _el(id)           { return document.getElementById(id); }
function _show(id, v='')   { const e = _el(id); if (e) e.style.display = v; }
function _hide(id)         { const e = _el(id); if (e) e.style.display = 'none'; }
function _text(id, t)      { const e = _el(id); if (e) e.textContent = t; }
function _html(id, h)      { const e = _el(id); if (e) e.innerHTML = h; }
function _width(id, w)     { const e = _el(id); if (e) e.style.width = w; }

/* ── Normal / Web Search mode: unified layout with think/AI-IO/steps ──────── */
async function _handleChatMode(query) {
  const input = _el('research-input');
  if (!input) return;
  input.value = '';
  autoResize(input);

  const isFirstTurn = _chatHistory.length === 0;

  /* ── On first turn: reset all panels (subsequent turns append) ────────── */
  if (isFirstTurn) {
    _hide('research-placeholder');
    _hide('report-output');
    _hide('report-actions');
    _html('step-list', '');
    _html('sources-list', '');
    _text('source-count', '');
    _width('progress-bar', '0%');
    resetThinkBlock();
    resetAioBlock();
  }

  _text('progress-label', 'Running…');

  /* Show chat history, hide the inner scroll wrapper (placeholder/report) */
  const chatHist    = _el('chat-history');
  const innerScroll = _el('research-content-inner');
  if (chatHist)    chatHist.style.display    = 'flex';
  if (innerScroll) innerScroll.style.display = 'none';

  /* ── Append user bubble ─────────────────────────────────────────────── */
  const userMsg = _mkBubble('user', query);
  chatHist.appendChild(userMsg);
  userMsg.scrollIntoView({ block: 'end', behavior: 'smooth' });

  /* ── Append empty AI bubble (stream into it) ─────────────────────────── */
  const aiMsg     = _mkBubble('ai', '');
  const aiContent = aiMsg.querySelector('.msg-content');
  aiContent.innerHTML = '<span class="pulse">▋</span>';
  chatHist.appendChild(aiMsg);
  aiMsg.scrollIntoView({ block: 'end', behavior: 'smooth' });

  /* ── Left-panel step: new turn ─────────────────────────────────────────── */
  const modeLabel = currentMode === 'websearch' ? '🌐 Web Search' : '💬 Normal';
  const turnNum   = Math.floor(_chatHistory.length / 2) + 1;
  addThinkStep(`─── Turn ${turnNum}: ${query.slice(0, 60)} ───`, 'plan');
  addStep(`${modeLabel} · Turn ${turnNum}`, 'active');
  _width('progress-bar', '0%');

  setBusy(true);
  isStreaming = true;

  let fullText    = '';
  let webSearched = false;
  const _startMs  = Date.now();

  try {
    const body = {
      message:    query,
      history:    _chatHistory.map(m => ({ role: m.role, content: m.content })),
      // Normal mode: force off (never auto-search). Web Search mode: force on.
      web_search: currentMode === 'websearch' ? true : currentMode === 'normal' ? false : null,
    };

    const res = await fetch(`${API}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
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
              aiContent.innerHTML = '<span class="chat-searching">🌐 Searching the web…</span>';
              addThinkStep('Searching the live web…', 'search');
              addStep('🌐 Searching the web…', 'active');
              _text('progress-label', 'Searching…');
              break;

            case 'search_result':
              /* Individual web result — show in crawl panel + source list */
              if (ev.url || ev.title) {
                const srcObj = {
                  url: ev.url || '',
                  title: ev.title || ev.url || '',
                  task: query,
                  credibility: ev.credibility || 0,
                };
                addSourceCard(srcObj);
                addThinkStep(`Source: ${ev.title || ev.url}`, 'store');
                const sc = _el('source-count');
                if (sc) {
                  const cur = parseInt(sc.textContent.replace(/\D/g,'')) || 0;
                  sc.textContent = ` (${cur + 1})`;
                }
              }
              break;

            case 'status':
              /* Backend status messages → think block + step list */
              addThinkStep(ev.message || ev.content || '', _thinkKind(ev.message || ''));
              break;

            case 'think':
              /* AI I/O debug cards */
              addAioCard(
                ev.role   || 'chat',
                ev.prompt || '',
                ev.output || ev.text || '',
                ev.think  || '',
              );
              break;

            case 'token':
              fullText += ev.content;
              aiContent.innerHTML = marked.parse(fullText);
              aiMsg.scrollIntoView({ block: 'end', behavior: 'smooth' });
              _text('progress-label', 'Generating…');
              break;

            case 'done': {
              const elapsedSec = ((Date.now() - _startMs) / 1000).toFixed(1);
              addStep('✅ Done', 'done');
              addThinkStep(`Done · ${elapsedSec}s`, 'done');
              finaliseThinkBlock(_thinkStepCount, 0, elapsedSec);
              _width('progress-bar', '100%');
              _text('progress-label', `Done · ${elapsedSec}s`);
              _addMsgActions(aiMsg, fullText, webSearched);
              break;
            }

            case 'error':
              aiContent.innerHTML = `<span style="color:var(--red)">Error: ${escHtml(ev.message)}</span>`;
              addThinkStep(`Error: ${ev.message}`, 'error');
              addStep(`❌ ${ev.message}`, 'error');
              break;
          }
        } catch { /* skip malformed */ }
      }
    }
  } catch (e) {
    aiContent.innerHTML = `<span style="color:var(--red)">Connection error: ${escHtml(e.message)}</span>`;
    addThinkStep(`Connection error: ${e.message}`, 'error');
    addStep(`❌ Connection error`, 'error');
  }

  /* Persist to history for next turn */
  _chatHistory.push({ role: 'user',      content: query    });
  _chatHistory.push({ role: 'assistant', content: fullText });

  setBusy(false);
  isStreaming = false;
}

/** Build a user or AI message bubble element */
function _mkBubble(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const avatar = role === 'user' ? '🧑' : '🤖';
  const label  = role === 'user' ? 'You'  : 'AI';
  wrap.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-role">${label}</div>
      <div class="msg-content">${role === 'user' ? escHtml(text) : (text ? marked.parse(text) : '')}</div>
    </div>`;
  return wrap;
}

/** Append copy button (and web badge) below an AI bubble */
function _addMsgActions(msgEl, text, webSearched) {
  // Remove the pulse cursor if still there
  const content = msgEl.querySelector('.msg-content');
  if (content && content.querySelector('.pulse')) {
    content.innerHTML = marked.parse(text);
  }
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  actions.style.display = 'flex';
  actions.innerHTML = `
    <button class="msg-action-btn" onclick="_copyMsg(this)" data-text="${escHtml(text).replace(/"/g,'&quot;')}">📋 Copy</button>
    ${webSearched ? '<span class="msg-web-badge">🌐 web-grounded</span>' : ''}`;
  msgEl.querySelector('.msg-body').appendChild(actions);
}

/** Copy text content of a message */
function _copyMsg(btn) {
  const text = btn.dataset.text || '';
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  });
}

/* ── Deep Research pipeline ───────────────────────────────────────────────── */
async function _handleDeepResearch(query) {
  const input = _el('research-input');
  if (!input) return;
  input.value = '';
  input.style.height = 'auto';

  /* ── Reset all UI panels ───────────────────────────────────────────── */
  _hide('research-placeholder');
  _hide('report-output');
  _html('report-output', '');
  _hide('report-actions');
  _html('step-list', '');
  _html('sources-list', '');
  _text('source-count', '');
  _width('progress-bar', '0%');
  _text('progress-label', 'Starting…');

  /* Hide chat history — this is deep research mode */
  _hide('chat-history');
  _show('research-content-inner', 'flex');

  const metaEl = _el('report-meta');
  if (metaEl) metaEl.textContent = '';
  _lastReportText  = '';
  _researchQuery   = query;
  resetKG();
  resetThinkBlock();
  resetAioBlock();

  /* Keep crawl panel closed until Phase 0 begins */
  const crawlPanel = _el('crawl-panel');
  if (crawlPanel) crawlPanel.classList.remove('open');
  _html('crawl-feed', '');
  _text('crawl-sites-n', '0');
  _text('crawl-phase-label', '');
  _crawl.phase0 = false;
  _crawl.phase1 = false;

  setBusy(true);
  setResearchBusy(true);

  /* ── Live stats tracker ─────────────────────────────────────────────── */
  // Estimates time remaining + AI queries left based on progress events.
  // Assumptions (conservative): 5 pages/task × 10 chunks/page × 1 AI call/chunk
  //   + 1 synthesis/page + 1 self-critique/task + 1 final report = ~58 calls/task
  const CALLS_PER_TASK   = 58;   // rough AI calls per research task
  const SECS_PER_CALL    = 18;   // ~18s per AI generation at 10-14 tok/s
  let   _totalTasks      = 0;
  let   _currentTask     = 0;
  let   _aiCallsDone     = 0;    // incremented on each status with chunk/analyse

  function _updateStatusBar(taskLabel) {
    const taskEl = document.getElementById('rsb-task');
    const qEl    = document.getElementById('rsb-q-val');
    const tEl    = document.getElementById('rsb-t-val');
    if (taskEl && taskLabel) taskEl.textContent = taskLabel;
    if (_totalTasks > 0) {
      const tasksLeft    = Math.max(0, _totalTasks - _currentTask);
      const queriesLeft  = tasksLeft * CALLS_PER_TASK + 1; // +1 for report
      const secsLeft     = queriesLeft * SECS_PER_CALL;
      if (qEl) qEl.textContent = queriesLeft.toLocaleString();
      if (tEl) tEl.textContent = _fmtTime(secsLeft);
    }
  }

  function _fmtTime(secs) {
    if (secs < 60)   return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.round(secs / 60)}m`;
    const h = Math.floor(secs / 3600);
    const m = Math.round((secs % 3600) / 60);
    return `${h}h ${m}m`;
  }

  /* Thought-process: first step */
  addThinkStep('Planning research…', 'step');
  addStep('🧠 Planning research…', 'active');

  let sourceCount = 0;
  const _startMs  = Date.now();

  try {
    const res = await fetch(`${API}/research`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, file_paths: uploadedFilePaths }),
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
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch { continue; }

        switch (ev.type) {

          case 'status': {
            /* Count AI calls for time estimate */
            const m = ev.message || '';
            if (m.includes('📝 Chunk') || m.includes('🔀 Synth') ||
                m.includes('🔄 Self-critique') || m.includes('📝 Writing report')) {
              _aiCallsDone++;
            }
            /* Update status bar label with the latest status message */
            if (m && !m.includes('Phase 0') && _currentTask > 0) {
              _updateStatusBar(`[${_currentTask}/${_totalTasks}] ${m.replace(/^[\s📝🧠🔀🔄✅⚠️❌📄]+/, '').slice(0, 60)}`);
            }
            /* Route to crawl panel first; only add to step list if not consumed */
            const consumed = _routeStatusToCrawl(ev.message);
            if (!consumed) addStep(ev.message, 'active');
            /* Always add to thought-process panel */
            addThinkStep(ev.message, _thinkKind(ev.message));
            break;
          }

          case 'plan':
            _totalTasks = ev.plan.length;
            _updateStatusBar(`Planning complete — ${_totalTasks} tasks`);
            addStep(`📋 Plan ready — ${ev.plan.length} tasks`, 'done');
            addThinkStep(`Plan ready — ${ev.plan.length} tasks`, 'plan');
            ev.plan.forEach((t, i) => {
              addStep(`${i + 1}. ${t.slice(0, 72)}`, 'active');
              addThinkStep(`${i + 1}. ${t.slice(0, 80)}`, 'search');
            });
            break;

          case 'progress': {
            _currentTask = ev.current;
            _updateStatusBar(`Task ${ev.current}/${ev.total}: ${(ev.task || '').slice(0, 55)}`);
            const pct = Math.round((ev.current / ev.total) * 80);
            _width('progress-bar', pct + '%');
            _text('progress-label', `Task ${ev.current} / ${ev.total}`);
            break;
          }

          case 'source':
            sourceCount++;
            _text('source-count', ` (${sourceCount})`);
            addSourceCard(ev.source);
            addThinkStep(
              `Source: ${ev.source.title || ev.source.url} (cred ${ev.source.credibility})`,
              'store',
            );
            break;

          case 'graph':
            updateKG(ev.graph);
            break;

          case 'think':
            /* ev = { type, role, prompt, output, think } */
            addAioCard(
              ev.role   || 'chat',
              ev.prompt || '',
              ev.output || ev.text || '',
              ev.think  || '',
            );
            break;

          case 'report': {
            _width('progress-bar', '100%');
            _text('progress-label', 'Complete ✅');
            addStep('✅ Report generated!', 'done');
            addThinkStep('Report generated ✅', 'done');

            const elapsedSec = ((Date.now() - _startMs) / 1000).toFixed(1);
            finaliseThinkBlock(_thinkStepCount, sourceCount, elapsedSec);

            const out = _el('report-output');
            if (out) {
              out.style.display = 'block';
              out.innerHTML     = marked.parse(ev.content);
            }
            _lastReportText = ev.content;

            if (metaEl) {
              const words = ev.content.trim().split(/\s+/).length;
              metaEl.textContent =
                `${words.toLocaleString()} words · ${sourceCount} sources · ${elapsedSec}s`;
            }

            _show('report-actions', 'flex');
            scrollBottom('research-content-inner');
            break;
          }

          case 'done':
            addStep('🎉 Research complete', 'done');
            break;

          case 'error':
            addStep(`❌ ${escHtml(ev.message)}`, 'error');
            addThinkStep(ev.message, 'error');
            break;
        }
      }
    }
  } catch (e) {
    addStep(`❌ Connection error: ${escHtml(e.message)}`, 'error');
    addThinkStep(`Connection error: ${e.message}`, 'error');
  }

  setBusy(false);
  setResearchBusy(false);
}

/* ── Map a status message string to a think-row kind icon ───────────────── */
function _thinkKind(msg) {
  if (!msg) return 'step';
  const m = msg.toLowerCase();
  if (m.includes('scraping') || m.includes('scrape'))   return 'scrape';
  if (m.includes('stored') || m.includes('storing'))    return 'store';
  if (m.includes('search') || m.includes('query'))      return 'search';
  if (m.includes('✅') || m.includes('complete'))       return 'done';
  if (m.includes('❌') || m.includes('error') || m.includes('failed')) return 'error';
  if (m.includes('ℹ') || m.includes('info'))            return 'info';
  return 'step';
}

/* ── Step list ───────────────────────────────────────────────────────────── */
function addStep(msg, state = 'active') {
  const list  = _el('step-list');
  if (!list) return;
  const icons = { active: '⏳', done: '✅', error: '❌' };

  /* Mark the previous active step as done */
  list.querySelectorAll('.step-item.active').forEach(el => {
    el.classList.replace('active', 'done');
    el.querySelector('.step-icon').textContent = '✅';
  });

  const li = document.createElement('li');
  li.className = `step-item ${state}`;
  li.innerHTML =
    `<span class="step-icon">${icons[state] || '⏳'}</span>` +
    `<span>${escHtml(msg)}</span>`;
  list.appendChild(li);
  li.scrollIntoView({ block: 'nearest' });
}

/* ── Source cards ────────────────────────────────────────────────────────── */
function addSourceCard(source) {
  if (!source.url) return;
  const list  = _el('sources-list');
  if (!list) return;
  const cred  = source.credibility || 0;
  const color = cred >= 70 ? 'var(--green)' : cred >= 45 ? 'var(--yellow)' : 'var(--red)';

  const card = document.createElement('div');
  card.className = 'source-card';
  card.innerHTML = `
    <a href="${escHtml(source.url)}" target="_blank" rel="noopener"
       title="${escHtml(source.url)}">${escHtml(source.title || source.url)}</a>
    <div class="src-task">${escHtml((source.task || '').slice(0, 65))}</div>
    <div class="cred-row">
      <div class="cred-bar-wrap">
        <div class="cred-bar" style="width:${cred}%;background:${color}"></div>
      </div>
      <span class="cred-label" style="color:${color}">${cred}</span>
    </div>`;
  list.appendChild(card);
}

/* ── Bootstrap: called here (last script) so every helper is available ──── */
// switchMode() needs: resetThinkBlock/resetAioBlock (report.js),
// addStep/addSourceCard (research.js), escHtml (utils.js) — all loaded above.
switchMode('normal');
