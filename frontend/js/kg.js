/* ── Knowledge Graph panel ───────────────────────────────────────────────── */

const TYPE_CSS = {
  company:    'type-company',
  person:     'type-person',
  technology: 'type-technology',
  product:    'type-product',
  location:   'type-location',
  concept:    'type-concept',
};

function hideKG() {
  document.getElementById('kg-panel').classList.add('hidden');
  document.getElementById('kg-show-btn').classList.add('visible');
}

function showKG() {
  document.getElementById('kg-panel').classList.remove('hidden');
  document.getElementById('kg-show-btn').classList.remove('visible');
}

function resetKG() {
  document.getElementById('kg-node-count').textContent = '0';
  document.getElementById('kg-edge-count').textContent = '0';
  document.getElementById('kg-node-list').innerHTML =
    '<div style="font-size:12px;color:var(--muted);text-align:center;padding:16px 0">' +
    'Entities will appear<br>as research runs…</div>';
  document.getElementById('kg-edge-list').innerHTML =
    '<div style="font-size:11px;color:var(--muted);text-align:center;padding:8px 0">' +
    'No relations yet</div>';
}

function updateKG(graph) {
  if (!graph) return;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];

  document.getElementById('kg-node-count').textContent = nodes.length;
  document.getElementById('kg-edge-count').textContent = edges.length;
  if (nodes.length === 0) return;

  // Top 25 by confidence descending
  const sorted   = [...nodes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const nodeList = document.getElementById('kg-node-list');
  nodeList.innerHTML = '';
  sorted.slice(0, 25).forEach(n => {
    const css = TYPE_CSS[n.type] || 'type-entity';
    const div = document.createElement('div');
    div.className = 'kg-node';
    div.title     = `${n.label} · ${n.type} · confidence ${n.confidence}`;
    div.innerHTML = `
      <span class="node-label">${escHtml(n.label)}</span>
      <span class="node-type">
        <span class="type-badge ${css}">${escHtml(n.type)}</span>
      </span>`;
    nodeList.appendChild(div);
  });

  if (edges.length === 0) return;

  const labelOf = {};
  nodes.forEach(n => { labelOf[n.id] = n.label; });

  const edgeList = document.getElementById('kg-edge-list');
  edgeList.innerHTML = '';
  edges.slice(0, 20).forEach(e => {
    const src = escHtml(labelOf[e.source] || e.source);
    const tgt = escHtml(labelOf[e.target] || e.target);
    const rel = escHtml(e.relation || 'related_to');
    const div = document.createElement('div');
    div.className = 'kg-edge';
    div.innerHTML = `<span class="e-src">${src}</span>
      <span class="e-rel">→ ${rel} →</span>
      <span class="e-tgt">${tgt}</span>`;
    edgeList.appendChild(div);
  });
}
