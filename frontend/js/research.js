/* ── Deep Research — SSE pipeline + step list + source cards ─────────────── */

function handleResearchKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startResearch(); }
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
            /* Route to crawl panel first; only add to step list if not consumed */
            const consumed = _routeStatusToCrawl(ev.message);
            if (!consumed) addStep(ev.message, 'active');
            /* Always add to thought-process panel */
            addThinkStep(ev.message, _thinkKind(ev.message));
            break;
          }

          case 'plan':
            addStep(`📋 Plan ready — ${ev.plan.length} tasks`, 'done');
            addThinkStep(`Plan ready — ${ev.plan.length} tasks`, 'plan');
            ev.plan.forEach((t, i) => {
              addStep(`${i + 1}. ${t.slice(0, 72)}`, 'active');
              addThinkStep(`${i + 1}. ${t.slice(0, 80)}`, 'search');
            });
            break;

          case 'progress': {
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
