/* ── File upload ─────────────────────────────────────────────────────────── */

let uploadedFilePaths = [];

function handleFileSelect(evt) {
  const files = Array.from(evt.target.files);
  if (!files.length) return;
  uploadFiles(files);
  evt.target.value = '';
}

async function uploadFiles(files) {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  try {
    addStep(`📎 Uploading ${files.length} file(s)…`, 'active');
    const r = await fetch(`${API}/upload`, { method: 'POST', body: form });
    const d = await r.json();
    d.uploaded.forEach(u => {
      uploadedFilePaths.push(u.path);
      addFileChip(u.filename, u.path);
    });
    addStep(`✅ Uploaded ${d.uploaded.length} file(s)`, 'done');
  } catch (e) {
    addStep(`❌ Upload failed: ${e.message}`, 'error');
  }
}

function addFileChip(name, path) {
  const list = document.getElementById('files-list');
  const chip = document.createElement('div');
  chip.className    = 'file-chip';
  chip.dataset.path = path;

  const nameSpan  = document.createElement('span');
  nameSpan.textContent = '📄 ' + name;

  const removeBtn = document.createElement('button');
  removeBtn.textContent = '×';
  removeBtn.addEventListener('click', () => removeFile(removeBtn, path));

  chip.appendChild(nameSpan);
  chip.appendChild(removeBtn);
  list.appendChild(chip);
}

function removeFile(btn, path) {
  uploadedFilePaths = uploadedFilePaths.filter(p => p !== path);
  btn.closest('.file-chip').remove();
}
