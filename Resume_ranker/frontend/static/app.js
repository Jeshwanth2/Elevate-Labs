/* ──────────────────────────────────────────
   ResumeIQ — Frontend App Logic
────────────────────────────────────────── */

const API_BASE = 'http://localhost:5000/api';

// ── State ──────────────────────────────────
let files = [];

// ── DOM refs ───────────────────────────────
const dropzone     = document.getElementById('dropzone');
const fileInput    = document.getElementById('fileInput');
const fileList     = document.getElementById('fileList');
const jobDesc      = document.getElementById('jobDesc');
const rankBtn      = document.getElementById('rankBtn');
const btnLoader    = document.getElementById('btnLoader');
const errorBox     = document.getElementById('errorBox');
const resultsPanel = document.getElementById('resultsPanel');
const overviewBar  = document.getElementById('overviewBar');
const candidateGrid= document.getElementById('candidateGrid');
const resultCount  = document.getElementById('resultCount');
const resultTimestamp = document.getElementById('resultTimestamp');
const clearBtn     = document.getElementById('clearBtn');
const modalBackdrop= document.getElementById('modalBackdrop');
const modal        = document.getElementById('modal');
const modalContent = document.getElementById('modalContent');
const modalClose   = document.getElementById('modalClose');

// ── File Handling ──────────────────────────

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => addFiles([...e.target.files]));

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});

function addFiles(newFiles) {
  const allowed = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
  const extMap  = { pdf: true, docx: true, txt: true };

  newFiles.forEach(f => {
    const ext = f.name.split('.').pop().toLowerCase();
    if (!extMap[ext]) return;
    if (files.some(x => x.name === f.name)) return; // dedupe
    files.push(f);
  });

  renderFileList();
  updateRankBtn();
}

function removeFile(index) {
  files.splice(index, 1);
  renderFileList();
  updateRankBtn();
}

function renderFileList() {
  fileList.innerHTML = '';
  files.forEach((f, i) => {
    const ext = f.name.split('.').pop().toLowerCase();
    const size = formatBytes(f.size);
    const div = document.createElement('div');
    div.className = 'file-item';
    div.innerHTML = `
      <div class="file-item-left">
        <span class="file-type-badge badge-${ext}">${ext}</span>
        <span class="file-name">${f.name}</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.75rem">
        <span class="file-size">${size}</span>
        <button class="file-remove" title="Remove" onclick="removeFile(${i})">✕</button>
      </div>
    `;
    fileList.appendChild(div);
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function updateRankBtn() {
  rankBtn.disabled = files.length === 0 || jobDesc.value.trim().length < 20;
}

jobDesc.addEventListener('input', updateRankBtn);

// ── Ranking ────────────────────────────────

rankBtn.addEventListener('click', async () => {
  hideError();
  setLoading(true);

  const formData = new FormData();
  formData.append('job_description', jobDesc.value.trim());
  files.forEach(f => formData.append('resumes', f));

  try {
    const response = await fetch(`${API_BASE}/rank`, {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || !data.success) {
      showError(data.error || 'An error occurred. Is the backend running?');
      return;
    }

    renderResults(data.results);
  } catch (err) {
    showError('Cannot reach the backend server. Run: python backend/app.py');
  } finally {
    setLoading(false);
  }
});

function setLoading(state) {
  rankBtn.classList.toggle('loading', state);
  rankBtn.disabled = state;
  if (state) {
    rankBtn.querySelector('.btn-text').textContent = 'Analysing…';
  } else {
    rankBtn.querySelector('.btn-text').textContent = 'Rank Resumes';
    updateRankBtn();
  }
}

// ── Results ────────────────────────────────

function renderResults(results) {
  resultsPanel.hidden = false;
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Meta
  resultCount.textContent = `${results.length} candidate${results.length !== 1 ? 's' : ''} ranked`;
  resultTimestamp.textContent = new Date().toLocaleTimeString();

  // Overview
  const avg   = (results.reduce((s, r) => s + r.score, 0) / results.length).toFixed(1);
  const top   = results[0]?.score.toFixed(1) ?? '–';
  const excel = results.filter(r => r.score >= 65).length;

  overviewBar.innerHTML = `
    <div class="ov-item"><div class="ov-value">${results.length}</div><div class="ov-label">Resumes</div></div>
    <div class="ov-item"><div class="ov-value">${top}%</div><div class="ov-label">Top Score</div></div>
    <div class="ov-item"><div class="ov-value">${avg}%</div><div class="ov-label">Avg Score</div></div>
    <div class="ov-item"><div class="ov-value">${excel}</div><div class="ov-label">Strong Fits</div></div>
  `;

  // Cards
  candidateGrid.innerHTML = '';
  results.forEach((r, i) => {
    const card = buildCandidateCard(r, i);
    candidateGrid.appendChild(card);
    // animate bar after render
    setTimeout(() => {
      const fill = card.querySelector('.score-bar-fill');
      if (fill) fill.style.width = r.score + '%';
    }, 100 + i * 80);
  });
}

function buildCandidateCard(r, i) {
  const gradeClass = gradeToClass(r.grade);
  const rankClass  = i < 3 ? `r${i + 1}` : 'rn';
  const topSkills  = r.matched_skills.slice(0, 4);
  const hasMissing = r.missing_skills.length > 0;

  const card = document.createElement('div');
  card.className = `candidate-card rank-${i + 1}`;
  card.innerHTML = `
    <div class="rank-badge ${rankClass}">#${r.rank}</div>
    <div class="card-main">
      <div class="candidate-name">${escHtml(r.name)}</div>
      <div class="candidate-meta">
        ${topSkills.map(s => `<span class="skill-tag">${s}</span>`).join('')}
        ${hasMissing ? `<span class="skill-tag missing">+${r.missing_skills.length} missing</span>` : ''}
      </div>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:0%"></div>
      </div>
    </div>
    <div class="score-side">
      <div class="score-value ${gradeClass}">${r.score}%</div>
      <div class="grade-label">${r.grade}</div>
    </div>
  `;
  card.addEventListener('click', () => openModal(r));
  return card;
}

// ── Modal ──────────────────────────────────

function openModal(r) {
  const gradeClass = gradeToClass(r.grade);
  modalContent.innerHTML = `
    <div class="modal-title">${escHtml(r.name)}</div>
    <div class="modal-score ${gradeClass}">${r.score}%</div>
    <div class="modal-grade">Rank #${r.rank} · ${r.grade} · ${r.word_count} words</div>

    <div class="modal-divider"></div>
    <div class="modal-section-title">Score Breakdown</div>

    ${buildBreakdownRow('TF-IDF Similarity',  r.tfidf_score)}
    ${buildBreakdownRow('Keyword Coverage',   r.keyword_score)}
    ${buildBreakdownRow('Experience Level',   r.experience_score)}
    ${buildBreakdownRow('Education Level',    r.education_score)}
    ${buildBreakdownRow('Section Coverage',   r.section_score)}

    <div class="modal-divider"></div>
    <div class="modal-section-title">Matched Skills (${r.matched_skills.length})</div>
    <div class="skill-tags-row">
      ${r.matched_skills.length
        ? r.matched_skills.map(s => `<span class="stag stag-match">✓ ${s}</span>`).join('')
        : '<span style="color:var(--text-muted);font-size:.82rem">No matched skills detected</span>'
      }
    </div>

    <div class="modal-divider"></div>
    <div class="modal-section-title">Missing Skills (${r.missing_skills.length})</div>
    <div class="skill-tags-row">
      ${r.missing_skills.length
        ? r.missing_skills.map(s => `<span class="stag stag-miss">✗ ${s}</span>`).join('')
        : '<span style="color:var(--accent3);font-size:.82rem">All key skills present 🎉</span>'
      }
    </div>
  `;

  modalBackdrop.hidden = false;

  // Animate bars
  setTimeout(() => {
    modalContent.querySelectorAll('.br-bar-fill').forEach(el => {
      el.style.width = el.dataset.width + '%';
    });
  }, 50);
}

function buildBreakdownRow(label, score) {
  return `
    <div class="breakdown-row">
      <div class="br-label">${label}</div>
      <div class="br-bar-track">
        <div class="br-bar-fill" data-width="${score}" style="width:0%"></div>
      </div>
      <div class="br-value">${score}%</div>
    </div>
  `;
}

modalClose.addEventListener('click', closeModal);
modalBackdrop.addEventListener('click', e => { if (e.target === modalBackdrop) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

function closeModal() {
  modalBackdrop.hidden = true;
}

// ── Clear ──────────────────────────────────

clearBtn.addEventListener('click', () => {
  resultsPanel.hidden = true;
  candidateGrid.innerHTML = '';
});

// ── Helpers ────────────────────────────────

function showError(msg) {
  errorBox.textContent = '⚠ ' + msg;
  errorBox.hidden = false;
}
function hideError() {
  errorBox.hidden = true;
}

function gradeToClass(grade) {
  const map = {
    'Excellent':     'excellent',
    'Good':          'good',
    'Average':       'average',
    'Below Average': 'below',
    'Poor':          'poor',
  };
  return map[grade] || 'average';
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}