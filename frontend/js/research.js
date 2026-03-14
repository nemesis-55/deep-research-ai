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

  /* ── Reset all UI panels ───────────────────────────────────────────── */
  document.getElementById('research-placeholder').style.display = 'none';
  document.getElementById('report-output').style.display        = 'none';
  document.getElementById('report-output').innerHTML            = '';
  document.getElementById('report-actions').style.display       = 'none';
  document.getElementById('step-list').innerHTML                = '';
  document.getElementById('sources-list').innerHTML             = '';
  document.getElementById('source-count').textContent          = '';
  document.getElementById('progress-bar').style.width          = '0%';
  document.getElementById('progress-label').textContent        = 'Starting…';
  const metaEl = document.getElementById('report-meta');
  if (metaEl) metaEl.textContent = '';
  _lastReportText  = '';
  _researchQuery   = query;
  resetKG();
  resetThinkBlock();
  resetAioBlock();

  /* Keep crawl panel closed until Phase 0 begins */
  document.getElementById('crawl-panel').classList.remove('open');
  document.getElementById('crawl-feed').innerHTML = '';
  document.getElementById('crawl-sites-n').textContent    = '0';
  document.getElementById('crawl-phase-label').textContent = '';
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
            document.getElementById('progress-bar').style.width  = pct + '%';
            document.getElementById('progress-label').textContent =
              `Task ${ev.current} / ${ev.total}`;
            break;
          }

          case 'source':
            sourceCount++;
            document.getElementById('source-count').textContent = ` (${sourceCount})`;
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
            document.getElementById('progress-bar').style.width   = '100%';
            document.getElementById('progress-label').textContent = 'Complete ✅';
            addStep('✅ Report generated!', 'done');
            addThinkStep('Report generated ✅', 'done');

            const elapsedSec = ((Date.now() - _startMs) / 1000).toFixed(1);
            finaliseThinkBlock(_thinkStepCount, sourceCount, elapsedSec);

            const out = document.getElementById('report-output');
            out.style.display = 'block';
            out.innerHTML     = marked.parse(ev.content);
            _lastReportText   = ev.content;

            /* Report metadata */
            if (metaEl) {
              const words = ev.content.trim().split(/\s+/).length;
              metaEl.textContent =
                `${words.toLocaleString()} words · ${sourceCount} sources · ${elapsedSec}s`;
            }

            document.getElementById('report-actions').style.display = 'flex';
            scrollBottom('research-content');
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
  const list  = document.getElementById('step-list');
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
  const list  = document.getElementById('sources-list');
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
