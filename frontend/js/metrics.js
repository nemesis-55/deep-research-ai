/* ── System Metrics Panel — Apple Silicon Mac Mini M4 ────────────────────── */

const _METRICS_INTERVAL_MS = 3000;
let   _metricsTimer        = null;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function _fmtBytes(b, decimals = 1) {
  if (b == null) return '—';
  if (b < 1024)           return b + ' B';
  if (b < 1024 ** 2)      return (b / 1024).toFixed(decimals) + ' KB';
  if (b < 1024 ** 3)      return (b / 1024 ** 2).toFixed(decimals) + ' MB';
  return (b / 1024 ** 3).toFixed(decimals) + ' GB';
}

function _fmtPct(v) {
  return v == null ? '—' : v.toFixed(1) + '%';
}

/** Fill a gauge: bar width + label text + colour by threshold. */
function _gauge(barId, labelId, pct, warnAt = 70, critAt = 90) {
  const bar   = document.getElementById(barId);
  const label = document.getElementById(labelId);
  if (!bar || !label) return;
  const p = pct ?? 0;
  bar.style.width = Math.min(p, 100) + '%';
  bar.className   = 'mbar-fill';
  if      (p >= critAt) bar.classList.add('crit');
  else if (p >= warnAt) bar.classList.add('warn');
  label.textContent = _fmtPct(pct);
}

/** Set a stat-cell value; dim if null. */
function _stat(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text ?? '—';
}

/* ── Render ──────────────────────────────────────────────────────────────── */

function _renderMetrics(d) {
  /* ── CPU ─────────────────────────────────────────────────── */
  const cpu = d.cpu || {};
  _gauge('m-cpu-bar', 'm-cpu-pct', cpu.pct, 70, 90);
  _stat('m-cpu-cores',  cpu.cores_physical != null
    ? `${cpu.cores_physical}P+${(cpu.cores_logical - cpu.cores_physical) || 0}E cores`
    : '—');
  _stat('m-cpu-freq',   cpu.freq_mhz != null ? (cpu.freq_mhz / 1000).toFixed(2) + ' GHz' : '—');
  _stat('m-load',       cpu.load1 != null
    ? `${cpu.load1.toFixed(2)} / ${cpu.load5.toFixed(2)} / ${cpu.load15.toFixed(2)}`
    : '—');

  /* Per-core mini-bars */
  const perCoreEl = document.getElementById('m-per-core');
  if (perCoreEl && Array.isArray(cpu.per_core)) {
    perCoreEl.innerHTML = cpu.per_core.map((p, i) => `
      <div class="core-bar-wrap" title="Core ${i}: ${p.toFixed(0)}%">
        <div class="core-bar-fill" style="height:${Math.min(p,100)}%"></div>
      </div>`).join('');
  }

  /* ── RAM ─────────────────────────────────────────────────── */
  const ram = d.ram || {};
  _gauge('m-ram-bar', 'm-ram-pct', ram.pct, 70, 88);
  _stat('m-ram-used',  `${_fmtBytes(ram.used_bytes)} / ${_fmtBytes(ram.total_bytes)}`);
  _stat('m-ram-avail', _fmtBytes(ram.available_bytes));
  _stat('m-swap-used', ram.swap_used_bytes != null
    ? `${_fmtBytes(ram.swap_used_bytes)} (${_fmtPct(ram.swap_pct)})` : '—');

  /* ── GPU (Apple Silicon unified) ────────────────────────── */
  const gpu = d.gpu || {};
  _gauge('m-gpu-bar', 'm-gpu-pct', gpu.util_pct, 65, 85);
  _stat('m-gpu-renderer', _fmtPct(gpu.renderer_pct));
  _stat('m-gpu-tiler',    _fmtPct(gpu.tiler_pct));
  _stat('m-gpu-vram-used',  _fmtBytes(gpu.vram_used_bytes));
  _stat('m-gpu-vram-alloc', _fmtBytes(gpu.vram_alloc_bytes));

  /* VRAM bar — fraction of total RAM consumed by GPU */
  _gauge('m-vram-bar', 'm-vram-pct', gpu.vram_used_pct, 40, 65);

  /* ── Disk ─────────────────────────────────────────────────── */
  const disk = d.disk || {};
  _gauge('m-disk-bar', 'm-disk-pct', disk.pct, 80, 92);
  _stat('m-disk-free',  _fmtBytes(disk.free_bytes));
  _stat('m-disk-total', _fmtBytes(disk.total_bytes));

  /* ── Process (this server) ───────────────────────────────── */
  const proc = d.process || {};
  _stat('m-proc-rss',     _fmtBytes(proc.rss_bytes));
  _stat('m-proc-cpu',     _fmtPct(proc.cpu_pct));
  _stat('m-proc-threads', proc.threads != null ? proc.threads + ' threads' : '—');

  /* Timestamp */
  const ts = document.getElementById('m-timestamp');
  if (ts) ts.textContent = 'Updated ' + new Date().toLocaleTimeString();
}

/* ── Fetch & update ──────────────────────────────────────────────────────── */

async function fetchMetrics() {
  try {
    const r = await fetch(`${API}/metrics`);
    if (!r.ok) return;
    const d = await r.json();
    if (d.error) return;
    _renderMetrics(d);
  } catch { /* backend not ready yet */ }
}

function startMetrics() {
  fetchMetrics();
  if (_metricsTimer) clearInterval(_metricsTimer);
  _metricsTimer = setInterval(fetchMetrics, _METRICS_INTERVAL_MS);
}

/* ── AI Performance Insights ─────────────────────────────────────────────── */

async function fetchInsights() {
  try {
    const r = await fetch(`${API}/metrics/insights`);
    if (!r.ok) return;
    const d = await r.json();
    _renderInsights(d);
  } catch { /* backend not ready */ }
}

function _renderInsights(d) {
  const empty  = document.getElementById('insights-empty');
  const body   = document.getElementById('insights-body');
  const tiers  = document.getElementById('ins-tiers');
  const recs   = document.getElementById('ins-recs');
  if (!body) return;

  if (!d.total_requests) {
    if (empty) empty.style.display = '';
    body.style.display = 'none';
    return;
  }
  if (empty) empty.style.display = 'none';
  body.style.display = '';

  const el = id => document.getElementById(id);
  if (el('ins-tps'))  el('ins-tps').textContent  = d.overall_tok_per_s + ' tok/s';
  if (el('ins-reqs')) el('ins-reqs').textContent = d.total_requests + ' calls';

  /* Per-tier breakdown */
  if (tiers) {
    const TIER_ICONS = { trivial: '⚡', conversational: '💬', technical: '🔬' };
    tiers.innerHTML = Object.entries(d.tier_stats || {}).map(([cx, s]) => `
      <div class="ins-tier">
        <span class="ins-tier-name">${TIER_ICONS[cx] || '•'} ${cx}</span>
        <span class="ins-tier-stats">
          ${s.tok_per_s_avg} tok/s · ${s.count} calls · ${s.avg_elapsed_s}s avg
        </span>
      </div>`).join('');
  }

  /* Optimisation recommendations */
  if (recs) {
    const recList = d.recommendations || [];
    if (!recList.length) {
      recs.innerHTML = '<div class="ins-rec ins-ok">✅ Token budgets optimal</div>';
    } else {
      recs.innerHTML = recList.map(r => `
        <div class="ins-rec">
          <span class="ins-rec-dir">${r.direction}</span>
          <span class="ins-rec-tier">${r.tier}</span>:
          ${r.current_budget} → <strong>${r.recommended}</strong> tokens
          <div class="ins-rec-reason">${escHtml(r.reason)}</div>
        </div>`).join('');
    }
  }
}

/* Auto-start on page load */
startMetrics();
/* Fetch insights once on load, then every 60 s */
setTimeout(() => {
  fetchInsights();
  setInterval(fetchInsights, 60_000);
}, 2000);
