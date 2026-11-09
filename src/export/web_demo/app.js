/**
 * Web Demo
 * Orchestrates 6 Machine Learning Strategies & Heuristic Baseline across 11 Languages.
 */

let evaluator = null;
let currentPieces = [];
let sampleDocs = {};
let benchmarkMatrix = {};

const STRATEGY_INFO = {
  'heuristic': {
    matrixKey: 'Heuristic Baseline',
    title: 'Heuristic Baseline (Zero ML)',
    badge: 'Baseline',
    desc: 'Rule-based heuristics using terminal punctuation and bounding-box vertical gap. Lacks layout nuance and fails on multi-line headings, quotes, and bullet continuations (59.57% Global Macro F1).'
  },
  'lang_gbdt': {
    matrixKey: 'Language-Specific GBDT: GBDT Only',
    title: 'Strategy 1.1: Language-Specific GBDT Only',
    badge: 'Monolingual GBDT',
    desc: 'Evaluates using a dedicated LightGBM GBDT trained solely on the chosen language. Suffers from vocabulary and layout sample fragmentation compared to the unified universal model (87.15% Global Macro F1).'
  },
  'lang_gbdt_lang_encoder': {
    matrixKey: 'Language-Specific GBDT: + Language-Specific Cross-Encoders',
    title: 'Strategy 1.2: Language GBDT + 11 Lang Cross-Encoders',
    badge: '2-Stage Specialized',
    desc: 'Routes ambiguous boundaries (0.10 < P < 0.90) to the language-specific transformer. Specialized fine-tuning on small pools (~3.5k samples) over-indexes on document quirks and incurs ~2.74ms latency across 4.8 GB of weights (90.39% Global Macro F1).'
  },
  'lang_gbdt_univ_encoder': {
    matrixKey: 'Language-Specific GBDT: + Universal Cross-Encoder',
    title: 'Strategy 1.3: Language GBDT + Universal Cross-Encoder',
    badge: 'Hybrid Cascade',
    desc: 'Combines the language GBDT fast-path with the Universal Cross-Encoder for ambiguous decisions. Beats 11 language-specific encoders by +1.63% F1 due to cross-lingual semantic transfer (92.02% Global Macro F1).'
  },
  'univ_gbdt': {
    matrixKey: 'Universal GBDT: GBDT Only',
    title: 'Strategy 2.1: Universal GBDT Only',
    badge: 'Production Champion',
    desc: 'The standalone Universal GBDT is the overall efficiency champion: 1.0 MB footprint, 13 µs latency, 92.20% Global Macro F1, and highest Phrase Exact Match (76.98%). Requires zero external neural runtimes.'
  },
  'univ_gbdt_lang_encoder': {
    matrixKey: 'Universal GBDT: + Language-Specific Cross-Encoders',
    title: 'Strategy 2.2: Universal GBDT + 11 Lang Cross-Encoders',
    badge: 'Specialization Penalty',
    desc: 'Pairing the Universal GBDT with monolingual cross-encoders degrades performance from 92.20% down to 91.60% (-0.60%) due to vocabulary fragmentation and out-of-vocabulary artifacts in educational PDFs.'
  },
  'univ_gbdt_univ_encoder': {
    matrixKey: 'Universal GBDT: + Universal Cross-Encoder',
    title: 'Strategy 2.3: Universal GBDT + Universal Cross-Encoder',
    badge: 'Highest Accuracy',
    desc: 'Universal 2-Stage Cascade achieves state-of-the-art accuracy (94.45%) and Global Macro F1 (92.83%). The GBDT resolves unambiguous decisions in 13 µs, and the Universal MiniLM arbitrates complex cases with 0.70ms average latency.'
  }
};

async function init() {
  evaluator = new GBDTEvaluator();
  const statusElem = document.getElementById('status-text');

  try {
    statusElem.innerText = 'Loading models & benchmark data...';

    // 1. Load Universal GBDT
    let univResp;
    try {
      univResp = await fetch('models/gbdt_universal.json');
    } catch {
      univResp = await fetch('gbdt_model.json');
    }
    const univJson = await univResp.json();
    evaluator.registerModel('universal', univJson);

    // 2. Load Benchmark Matrix
    try {
      const bResp = await fetch('benchmark_matrix.json');
      benchmarkMatrix = await bResp.json();
    } catch (e) {
      console.warn('Could not load benchmark_matrix.json, using fallback stats:', e);
    }

    // 3. Load Multilingual Samples
    try {
      const sResp = await fetch('sample_docs.json');
      sampleDocs = await sResp.json();
    } catch (e) {
      const fallbackResp = await fetch('sample_doc.json');
      sampleDocs = { pt: await fallbackResp.json() };
    }

    // 4. Preload Portuguese model
    await loadLanguageModel('pt');

    statusElem.innerText = 'Models Ready (Universal GBDT + Multilingual Portfolio)';
    setSampleDoc('pt');
  } catch (err) {
    console.error('Initialization error:', err);
    statusElem.innerText = 'Error: ' + err.message;
  }
}

async function loadLanguageModel(lang) {
  if (evaluator.models[lang]) return;
  try {
    const resp = await fetch(`models/gbdt_${lang}.json`);
    if (resp.ok) {
      const json = await resp.json();
      evaluator.registerModel(lang, json);
    }
  } catch (e) {
    console.warn(`Could not load dedicated model for ${lang}, using universal model:`, e);
  }
}

async function onLanguageChange() {
  const lang = document.getElementById('lang-select').value;
  await loadLanguageModel(lang);

  const sampleSelect = document.getElementById('sample-select').value;
  if (sampleSelect === 'current') {
    setSampleDoc(lang);
  } else {
    runInference();
  }
}

function onSampleChange() {
  const choice = document.getElementById('sample-select').value;
  const lang = choice === 'current' ? document.getElementById('lang-select').value : choice;
  setSampleDoc(lang);
}

let viewMode = 'side-by-side'; // 'side-by-side' or 'split'
let lastDecisions = [];
let lastPhrases = [];

function setViewMode(mode) {
  viewMode = mode;
  document.getElementById('btn-side-by-side').classList.toggle('active', mode === 'side-by-side');
  document.getElementById('btn-split').classList.toggle('active', mode === 'split');
  const comp = document.getElementById('comparison-view');
  if (comp) {
    comp.className = `comparison-container mode-${mode}`;
  }
  renderComparisonView();
}

function setSampleDoc(lang) {
  currentPieces = sampleDocs[lang] || sampleDocs['pt'] || [];
  runInference();
}

function renderStrategyCard(strategyKey, lang) {
  const info = STRATEGY_INFO[strategyKey] || STRATEGY_INFO['univ_gbdt'];
  document.getElementById('strategy-card-title').innerText = info.title;
  document.getElementById('strategy-card-badge').innerText = info.badge;
  document.getElementById('strategy-card-desc').innerText = info.desc;

  const matrixEntry = benchmarkMatrix[info.matrixKey] || {};
  const globalM = matrixEntry['GLOBAL'] || {};
  const langM = matrixEntry[lang] || {};

  document.getElementById('metric-global-f1').innerText = globalM.macro_f1 ? `${(globalM.macro_f1 * 100).toFixed(2)}%` : '--';
  document.getElementById('metric-lang-label').innerText = `${lang.toUpperCase()} Macro F1`;
  document.getElementById('metric-lang-f1').innerText = langM.macro_f1 ? `${(langM.macro_f1 * 100).toFixed(2)}%` : '--';
  document.getElementById('metric-global-acc').innerText = globalM.accuracy ? `${(globalM.accuracy * 100).toFixed(2)}%` : '--';
  document.getElementById('metric-global-em').innerText = globalM.phrase_em ? `${(globalM.phrase_em * 100).toFixed(2)}%` : '--';
  document.getElementById('metric-latency').innerText = globalM.latency_ms ? `${globalM.latency_ms.toFixed(3)} ms` : '--';
}

function runInference() {
  if (!evaluator || currentPieces.length < 1) return;

  const strategyKey = document.getElementById('strategy-select').value;
  const lang = document.getElementById('lang-select').value;

  renderStrategyCard(strategyKey, lang);

  const startTime = performance.now();
  let joinCount = 0;
  let splitCount = 0;
  let fastPathCount = 0;

  const decisions = [];

  for (let i = 0; i < currentPieces.length - 1; i++) {
    const pA = currentPieces[i];
    const pB = currentPieces[i + 1];

    const feats = extractBoundaryFeaturesJS(pA, pB);
    const result = evaluator.evaluateBoundary(strategyKey, lang, feats, pA, pB);
    decisions.push(result);

    if (result.decision === 1) joinCount++; else splitCount++;
    if (!result.route.includes('Stage 2')) {
      fastPathCount++;
    }
  }

  const elapsed = performance.now() - startTime;
  const avgLatency = (currentPieces.length > 1) ? (elapsed / (currentPieces.length - 1)) : 0;

  const fastPathPct = currentPieces.length > 1
    ? ((fastPathCount / (currentPieces.length - 1)) * 100).toFixed(0)
    : '100';
  document.getElementById('stat-pieces').innerText = currentPieces.length;
  document.getElementById('stat-boundaries').innerText = Math.max(0, currentPieces.length - 1);
  document.getElementById('stat-fast-path').innerText = `${fastPathPct}%`;
  document.getElementById('stat-latency').innerText = `${avgLatency.toFixed(3)} ms`;

  lastDecisions = decisions;
  buildPhrases(decisions);
  renderComparisonView();
}

function buildPhrases(decisions) {
  const phrases = [];
  if (currentPieces.length === 0) {
    lastPhrases = [];
    return;
  }

  let currentPhrase = {
    id: 0,
    page: currentPieces[0].page_num !== undefined ? currentPieces[0].page_num : (currentPieces[0].page || 0),
    pieces: [currentPieces[0].text || ''],
    pieceIndices: [0]
  };

  for (let i = 0; i < decisions.length; i++) {
    const dec = decisions[i];
    if (dec.decision === 1) {
      currentPhrase.pieces.push(currentPieces[i + 1].text || '');
      currentPhrase.pieceIndices.push(i + 1);
    } else {
      phrases.push(currentPhrase);
      currentPhrase = {
        id: phrases.length,
        page: currentPieces[i + 1].page_num !== undefined ? currentPieces[i + 1].page_num : (currentPieces[i + 1].page || 0),
        pieces: [currentPieces[i + 1].text || ''],
        pieceIndices: [i + 1]
      };
    }
  }
  phrases.push(currentPhrase);
  lastPhrases = phrases;
  document.getElementById('stat-phrases').innerText = phrases.length;
}

function renderComparisonView() {
  const container = document.getElementById('comparison-view');
  if (!container) return;
  container.innerHTML = '';

  if (viewMode === 'side-by-side') {
    renderAlignedRowsView(container);
  } else {
    renderSplitColumnsView(container);
  }
}

function renderAlignedRowsView(container) {
  // Render sticky comparison header
  const header = document.createElement('div');
  header.className = 'aligned-header-row';
  header.innerHTML = `
    <div class="aligned-header-cell">
      <span>PDF Marked Content Pieces (Input Stream)</span>
      <span class="panel-badge">Fragment Stream</span>
    </div>
    <div class="aligned-header-cell">
      <span>Combined Text Segment (Output Segment)</span>
      <span class="panel-badge">Navigable Target</span>
    </div>
  `;
  container.appendChild(header);

  lastPhrases.forEach((phrase, phraseIdx) => {
    const row = document.createElement('div');
    row.className = 'aligned-row';
    row.id = `aligned-row-${phraseIdx}`;
    row.setAttribute('data-phrase-id', phraseIdx);

    // Left side: Pieces and boundaries belonging to this phrase
    const piecesCol = document.createElement('div');
    piecesCol.className = 'aligned-pieces-col';

    phrase.pieceIndices.forEach((pIdx, idxInPhrase) => {
      const p = currentPieces[pIdx];
      const pCard = document.createElement('div');
      pCard.className = 'mc-piece-card';
      pCard.id = `piece-${pIdx}`;
      pCard.setAttribute('data-phrase-id', phraseIdx);
      pCard.setAttribute('data-piece-idx', pIdx);
      pCard.innerHTML = `
        <div class="mc-meta">
          <span class="piece-badge">Piece #${pIdx} &bull; Page ${p.page_num !== undefined ? p.page_num : (p.page || 0)}</span>
          <span class="font-badge">${p.font_name || p.fontName || 'Normal'} (${p.font_size || p.fontSize || 10}pt${p.is_bold ? ' &bull; Bold' : ''})</span>
          <span class="bbox-badge">BBox: [${(p.bbox || [0,0,0,0]).map(v => Math.round(v)).join(', ')}]</span>
        </div>
        <div class="mc-text">${escapeHtml(p.text || '')}</div>
      `;
      piecesCol.appendChild(pCard);

      // If there is an internal boundary between pieces inside this phrase
      if (idxInPhrase < phrase.pieceIndices.length - 1) {
        const bIdx = pIdx;
        const dec = lastDecisions[bIdx];
        if (dec) {
          const bDiv = document.createElement('div');
          bDiv.className = 'boundary-indicator boundary-join';
          bDiv.id = `boundary-${bIdx}`;
          bDiv.innerHTML = `
            <div class="boundary-top">
              <span class="boundary-label">
                ↦ ↤ JOIN (Continue)
                <span class="boundary-prob">(P=${(dec.confidence * 100).toFixed(1)}%)</span>
              </span>
              <span class="boundary-route-pill">${dec.route}</span>
            </div>
            <div class="boundary-reason">${escapeHtml(dec.reason)}</div>
          `;
          piecesCol.appendChild(bDiv);
        }
      }
    });

    // Right side: The reconstructed phrase card spanning the exact height of its constituent pieces
    const segmentCol = document.createElement('div');
    segmentCol.className = 'aligned-segment-col';

    const card = document.createElement('div');
    card.className = 'phrase-card';
    card.id = `phrase-card-${phraseIdx}`;
    card.setAttribute('data-phrase-id', phraseIdx);
    card.innerHTML = `
      <div class="phrase-header">
        <span class="phrase-badge">Segment #${phraseIdx + 1}</span>
        <span class="phrase-meta">Page ${phrase.page} &bull; ${phrase.pieces.length} Piece${phrase.pieces.length > 1 ? 's' : ''} Combined</span>
      </div>
      <div class="phrase-body">${escapeHtml(phrase.pieces.join(' '))}</div>
      <div class="phrase-footer">
        <span class="tag-pieces">Pieces: ${phrase.pieceIndices.map(i => `#${i}`).join(' + ')}</span>
        <span class="tag-action">Click to preview interactive anchor</span>
      </div>
    `;

    card.onclick = () => {
      alert(`Selected Text Segment #${phraseIdx + 1}:\n\n"${phrase.pieces.join(' ')}"\n\nCombined from Pieces: ${phrase.pieceIndices.map(i => `#${i}`).join(', ')}`);
    };

    segmentCol.appendChild(card);

    row.appendChild(piecesCol);
    row.appendChild(segmentCol);
    container.appendChild(row);

    // If this is NOT the last phrase, display the SPLIT boundary separating this row from the next
    const lastPieceOfPhrase = phrase.pieceIndices[phrase.pieceIndices.length - 1];
    if (lastPieceOfPhrase < currentPieces.length - 1) {
      const dec = lastDecisions[lastPieceOfPhrase];
      const sep = document.createElement('div');
      sep.className = 'aligned-split-separator';
      sep.id = `boundary-${lastPieceOfPhrase}`;
      sep.innerHTML = `
        <div class="split-divider-line"></div>
        <div class="split-badge-content">
          <span class="boundary-label">
            ↤ ↦ SPLIT &bull; Paragraph Break Between Piece #${lastPieceOfPhrase} and #${lastPieceOfPhrase + 1}
            ${dec ? `<span class="boundary-prob">(P=${(dec.confidence * 100).toFixed(1)}%)</span>` : ''}
          </span>
          ${dec ? `<span class="boundary-route-pill">${dec.route}</span>` : ''}
          ${dec ? `<span class="boundary-split-reason">${escapeHtml(dec.reason)}</span>` : ''}
        </div>
        <div class="split-divider-line"></div>
      `;
      container.appendChild(sep);
    }
  });

  setupHoverSync();
}

function renderSplitColumnsView(container) {
  const splitView = document.createElement('div');
  splitView.className = 'split-view';

  // Left panel: Pieces
  const leftPanel = document.createElement('div');
  leftPanel.className = 'panel';
  leftPanel.innerHTML = `
    <h2>
      <span>PDF Marked Content Pieces</span>
      <span class="panel-badge">Input Stream</span>
    </h2>
    <div id="pieces-container"></div>
  `;

  // Right panel: Phrases
  const rightPanel = document.createElement('div');
  rightPanel.className = 'panel';
  rightPanel.innerHTML = `
    <h2>
      <span>Combined Text Segments</span>
      <span class="panel-badge">Accessible Segments</span>
    </h2>
    <div id="phrases-container"></div>
  `;

  splitView.appendChild(leftPanel);
  splitView.appendChild(rightPanel);
  container.appendChild(splitView);

  const pContainer = leftPanel.querySelector('#pieces-container');
  const phContainer = rightPanel.querySelector('#phrases-container');

  currentPieces.forEach((p, idx) => {
    // Find which phrase this piece belongs to
    const phraseIdx = lastPhrases.findIndex(ph => ph.pieceIndices.includes(idx));

    const div = document.createElement('div');
    div.className = 'mc-piece-card';
    div.id = `piece-${idx}`;
    div.setAttribute('data-phrase-id', phraseIdx);
    div.setAttribute('data-piece-idx', idx);
    div.innerHTML = `
      <div class="mc-meta">
        <span class="piece-badge">Piece #${idx} (Page ${p.page_num !== undefined ? p.page_num : (p.page || 0)})</span>
        <span>${p.font_name || p.fontName || 'Normal'} (${p.font_size || p.fontSize || 10}pt)</span>
        <span>BBox: [${(p.bbox || [0,0,0,0]).map(v => Math.round(v)).join(', ')}]</span>
      </div>
      <div class="mc-text">${escapeHtml(p.text || '')}</div>
    `;
    pContainer.appendChild(div);

    if (idx < currentPieces.length - 1) {
      const dec = lastDecisions[idx];
      const isJoin = dec ? dec.decision === 1 : false;
      const bDiv = document.createElement('div');
      bDiv.className = `boundary-indicator ${isJoin ? 'boundary-join' : 'boundary-split'}`;
      bDiv.id = `boundary-${idx}`;
      bDiv.innerHTML = `
        <div class="boundary-top">
          <span class="boundary-label">
            ${isJoin ? '↦ ↤ JOIN (Continue)' : '↤ ↦ SPLIT (New Block)'}
            <span class="boundary-prob">(P=${dec ? (dec.confidence * 100).toFixed(1) : 0}%)</span>
          </span>
          <span class="boundary-route-pill">${dec ? dec.route : ''}</span>
        </div>
        <div class="boundary-reason">${escapeHtml(dec ? dec.reason : '')}</div>
      `;
      pContainer.appendChild(bDiv);
    }
  });

  lastPhrases.forEach((ph, idx) => {
    const card = document.createElement('div');
    card.className = 'phrase-card';
    card.id = `phrase-card-${idx}`;
    card.setAttribute('data-phrase-id', idx);
    card.innerHTML = `
      <div class="phrase-header">
        <span class="phrase-badge">Segment #${idx + 1}</span>
        <span>Page ${ph.page} &bull; ${ph.pieces.length} Piece${ph.pieces.length > 1 ? 's' : ''} Combined</span>
      </div>
      <div class="phrase-body">${escapeHtml(ph.pieces.join(' '))}</div>
      <div class="phrase-footer">
        <span class="tag-pieces">Pieces: ${ph.pieceIndices.map(i => `#${i}`).join(' + ')}</span>
      </div>
    `;
    card.onclick = () => {
      alert(`Selected Text Segment #${idx + 1}:\n\n"${ph.pieces.join(' ')}"\n\nCombined from Pieces: ${ph.pieceIndices.map(i => `#${i}`).join(', ')}`);
    };
    phContainer.appendChild(card);
  });

  setupHoverSync();
}

function setupHoverSync() {
  const pieceCards = document.querySelectorAll('.mc-piece-card');
  const phraseCards = document.querySelectorAll('.phrase-card');
  const alignedRows = document.querySelectorAll('.aligned-row');

  pieceCards.forEach(card => {
    const phraseId = card.getAttribute('data-phrase-id');
    card.addEventListener('mouseenter', () => highlightGroup(phraseId, true));
    card.addEventListener('mouseleave', () => highlightGroup(phraseId, false));
  });

  phraseCards.forEach(card => {
    const phraseId = card.getAttribute('data-phrase-id');
    card.addEventListener('mouseenter', () => highlightGroup(phraseId, true));
    card.addEventListener('mouseleave', () => highlightGroup(phraseId, false));
  });

  alignedRows.forEach(row => {
    const phraseId = row.getAttribute('data-phrase-id');
    row.addEventListener('mouseenter', () => highlightGroup(phraseId, true));
    row.addEventListener('mouseleave', () => highlightGroup(phraseId, false));
  });
}

function highlightGroup(phraseId, active) {
  if (phraseId === null || phraseId === undefined) return;
  const elements = document.querySelectorAll(`[data-phrase-id="${phraseId}"]`);
  elements.forEach(el => {
    if (active) {
      el.classList.add('highlight-match');
    } else {
      el.classList.remove('highlight-match');
    }
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}

window.onload = init;
