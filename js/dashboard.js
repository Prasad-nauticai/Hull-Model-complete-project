/* ═══════════════════════════════════════════════
   NautiCAI — Dashboard JavaScript
   Real AI Detection via FastAPI + YOLOv8
═══════════════════════════════════════════════ */

const API_BASE = 'http://127.0.0.1:8000';
window.API_BASE = API_BASE;
console.log('[NautiCAI] API_BASE =', API_BASE);



/* Class colour map — covers all classes from all 3 models */
const CLASS_COLORS = {
    'corrosion':   '#e74c3c',   // red
    'biofouling':  '#f0a500',   // amber
    'debris':      '#27ae60',   // green
    'anode':       '#00d4e8',   // cyan
    'paint_peel':  '#e74c3c',   // red
    'anomaly':     '#9b59b6',   // purple
};
const DEFAULT_COLOR = '#7a9eb5';

/* Normalize class name — unify across models */
function normalizeClassName(name) {
    if (!name) return name;
    const k = name.toLowerCase().trim();
    if (k === 'hull')          return 'corrosion';
    if (k === 'marine_growth') return 'biofouling';
    return k;
}

function classColor(name) {
    return CLASS_COLORS[normalizeClassName((name || '').toLowerCase().trim())] || DEFAULT_COLOR;
}

/* ── Upload drag-and-drop ─────────────────────── */
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const folderInput = document.getElementById('folder-input');
let uploadedFile = null;       // single‑file compat
let uploadedFiles = [];        // batch array
let uploadMode = 'image';      // 'image' | 'video' | 'folder'

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    handleFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', (e) => {
    handleFiles(Array.from(e.target.files));
});
folderInput.addEventListener('change', (e) => {
    const allFiles = Array.from(e.target.files);
    // Filter to images only when selecting a folder
    const imageFiles = allFiles.filter(f => f.type.startsWith('image/'));
    handleFiles(imageFiles);
});

/* Set upload type — each button directly opens its picker */
window.setUploadType = function (type) {
    uploadMode = type;
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    const btnId = type === 'folder' ? 'btn-folder' : (type === 'video' ? 'btn-video' : 'btn-image');
    document.getElementById(btnId)?.classList.add('active');

    if (type === 'folder') {
        folderInput.value = '';
        folderInput.click();
    } else if (type === 'video') {
        fileInput.accept = 'video/*';
        fileInput.removeAttribute('multiple');
        fileInput.value = '';
        fileInput.click();
    } else {
        fileInput.accept = 'image/*';
        fileInput.setAttribute('multiple', '');
        fileInput.value = '';
        fileInput.click();
    }
};

/* Upload zone click → open the picker matching current mode */
let _pickerOpen = false;
function openPicker() {
    if (_pickerOpen) return;
    _pickerOpen = true;
    setTimeout(() => { _pickerOpen = false; }, 800);

    if (uploadMode === 'folder') {
        folderInput.value = '';
        folderInput.click();
    } else {
        fileInput.value = '';
        fileInput.click();
    }
}

uploadZone.addEventListener('click', (e) => {
    // Ignore clicks on the file inputs themselves or the browse link
    if (e.target === fileInput || e.target === folderInput) return;
    if (e.target.id === 'upload-link-btn') return;
    openPicker();
});

/* "browse" link */
document.getElementById('upload-link-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    openPicker();
});

/* Handle incoming files (single or multiple) */
function handleFiles(files) {
    if (!files || files.length === 0) return;

    uploadedFiles = files;
    uploadedFile = files[0];  // keep single-file compat

    const batchBadge = document.getElementById('batch-badge');
    const batchCount = document.getElementById('batch-count');

    if (files.length > 1) {
        batchBadge.style.display = 'block';
        batchCount.textContent = files.length;
    } else {
        batchBadge.style.display = 'none';
    }

    // Show first file preview
    const f = files[0];
    const previewImg = document.getElementById('preview-image');
    const previewVid = document.getElementById('preview-video');
    const placeholder = document.getElementById('preview-placeholder');

    placeholder.style.display = 'none';

    if (f.type.startsWith('image/')) {
        previewImg.src = URL.createObjectURL(f);
        previewImg.style.display = 'block';
        previewVid.style.display = 'none';
    } else if (f.type.startsWith('video/')) {
        previewVid.src = URL.createObjectURL(f);
        previewVid.style.display = 'block';
        previewImg.style.display = 'none';
    }

    // Visual upload confirmation
    uploadZone.style.borderColor = 'var(--teal)';
    uploadZone.style.background = 'rgba(0,200,176,0.06)';
    setTimeout(() => {
        uploadZone.style.borderColor = '';
        uploadZone.style.background = '';
    }, 2000);
}

/* Keep old handleFile for compat */
function handleFile(file) { handleFiles([file]); }

/* ── Run AI Detection — calls real FastAPI ────── */
window.runAIDetection = async function () {
    if (uploadedFiles.length === 0 && !uploadedFile) {
        showToast('Please upload an image, video, or folder first.', 'warn');
        return;
    }

    const files = uploadedFiles.length > 0 ? uploadedFiles : [uploadedFile];
    const isBatch = files.length > 1;
    const isVideo = files.length === 1 && files[0].type.startsWith('video/');

    const emptyState   = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsEl    = document.getElementById('results-content');

    emptyState.style.display   = 'none';
    resultsEl.style.display    = 'none';
    loadingState.style.display = 'flex';

    const progressBar = document.getElementById('progress-bar');
    const stageLabel  = document.querySelector('.progress-stage') || { textContent: '' };
    progressBar.style.width = '0%';

    try {
        let mergedData = null;

        if (isVideo) {
            /* ── Single video: submit job, poll for live progress ── */
            stageLabel.textContent = 'Uploading video...';
            progressBar.style.width = '5%';

            const formData = new FormData();
            formData.append('file', files[0]);

            const resp = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const jobResp = await resp.json();

            // If backend returned a job_id it's async (video) — poll for progress
            if (jobResp.job_id) {
                mergedData = await _pollVideoJob(jobResp.job_id, progressBar, stageLabel);
            } else {
                // Synchronous response (shouldn't happen for video but handle gracefully)
                mergedData = jobResp;
            }

        } else if (isBatch) {
            /* ── Multiple images: process sequentially ── */
            const allDetections     = [];
            const allAnnotatedImages = [];
            let lastMetrics = null;
            let totalInfMs  = 0;
            let inspId      = '';

            for (let i = 0; i < files.length; i++) {
                const pct = Math.round(((i + 1) / files.length) * 95);
                progressBar.style.width = pct + '%';
                stageLabel.textContent  = `Analyzing ${i + 1} of ${files.length}: ${files[i].name}`;

                const formData = new FormData();
                formData.append('file', files[i]);

                const response = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
                if (!response.ok) {
                    const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
                    console.warn(`File ${files[i].name} failed:`, err.detail);
                    continue;
                }

                const data = await response.json();
                const fileDetections = data.detections || [];
                allDetections.push(...fileDetections);
                allAnnotatedImages.push({
                    filename:   files[i].name,
                    image:      data.annotated_image || null,
                    detections: fileDetections,
                    summary:    data.summary || null,
                });
                lastMetrics = data.model_metrics || lastMetrics;
                inspId      = data.inspection_id || inspId;
                totalInfMs += data.summary?.inference_time_ms || 0;
            }

            const maxConf = allDetections.length ? Math.max(...allDetections.map(d => d.confidence)) : 0;
            const avgConf = allDetections.length
                ? allDetections.reduce((a, d) => a + d.confidence, 0) / allDetections.length : 0;
            // Use backend risk thresholds (conf > 0.40 HIGH, > 0.25 MEDIUM, else LOW)
            const risk = maxConf > 0.40 ? 'HIGH' : maxConf > 0.25 ? 'MEDIUM' : maxConf > 0 ? 'LOW' : 'SAFE';

            mergedData = {
                inspection_id:    inspId || 'BATCH-' + Date.now(),
                detections:       allDetections,
                annotated_image:  allAnnotatedImages[0]?.image || null,
                annotated_images: allAnnotatedImages,
                summary: {
                    total:             allDetections.length,
                    risk_level:        risk,
                    avg_confidence:    Math.round(avgConf * 10000) / 10000,
                    inference_time_ms: totalInfMs,
                    files_processed:   files.length,
                },
                model_metrics: lastMetrics || { precision: null, recall: null, map50: null, map5095: null },
                timestamp:     new Date().toISOString(),
            };

        } else {
            /* ── Single image ── */
            const stages = [
                { pct: 15, label: 'Uploading file...' },
                { pct: 40, label: 'Preprocessing image...' },
                { pct: 65, label: 'Running YOLOv8 inference...' },
                { pct: 85, label: 'Classifying species...' },
                { pct: 95, label: 'Post-processing results...' },
            ];
            let stageIdx = 0;
            const stageTimer = setInterval(() => {
                if (stageIdx < stages.length) {
                    progressBar.style.width = stages[stageIdx].pct + '%';
                    stageLabel.textContent  = stages[stageIdx].label;
                    stageIdx++;
                }
            }, 600);

            const formData = new FormData();
            formData.append('file', files[0]);

            const response = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
            clearInterval(stageTimer);

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }
            mergedData = await response.json();
        }

        progressBar.style.width = '100%';
        stageLabel.textContent  = isVideo
            ? `Video analysis complete! ${mergedData.summary?.frames_analyzed || 0} frames, ${mergedData.detections?.length || 0} detections.`
            : isBatch
                ? `Batch complete! ${files.length} files, ${mergedData.detections.length} detections.`
                : 'Analysis complete!';
        await sleep(400);

        loadingState.style.display = 'none';
        showResults(mergedData);

    } catch (err) {
        loadingState.style.display = 'none';
        emptyState.style.display   = 'flex';
        progressBar.style.width    = '0%';

        if (err.message.includes('fetch') || err.message.includes('Failed')) {
            showToast(`API offline at ${API_BASE} — showing demo results. Start backend and retry.`, 'warn');
            showDemoResults();
        } else {
            showToast(`Detection error: ${err.message}`, 'error');
        }
    }
};

/* ── Poll async video job until done ─────────────── */
async function _pollVideoJob(jobId, progressBar, stageLabel) {
    const POLL_MS   = 2000;   // poll every 2 seconds
    const TIMEOUT_S = 1800;   // give up after 30 min
    const started   = Date.now();

    stageLabel.textContent  = 'Video queued — waiting for worker...';
    progressBar.style.width = '8%';

    while (true) {
        if ((Date.now() - started) / 1000 > TIMEOUT_S) {
            throw new Error('Video processing timed out after 30 minutes.');
        }

        await sleep(POLL_MS);

        let status;
        try {
            const r = await fetch(`${API_BASE}/job/${jobId}/status`);
            if (!r.ok) throw new Error(`Poll failed: HTTP ${r.status}`);
            status = await r.json();
        } catch (e) {
            console.warn('[poll] fetch error:', e.message);
            continue;
        }

        const pct = status.progress_pct || 0;
        // Reserve 8–95% for actual processing, 5% for upload, 5% for render
        progressBar.style.width = Math.max(8, Math.min(95, 8 + pct * 0.87)) + '%';

        if (status.status === 'running' || status.status === 'queued') {
            const frameInfo = status.current_file || 'processing...';
            const elapsed   = status.elapsed_sec ? ` (${status.elapsed_sec}s)` : '';
            stageLabel.textContent = `Analyzing video: ${frameInfo}${elapsed}`;
        }

        if (status.status === 'done') {
            if (!status.results) throw new Error('Job done but no results returned.');
            return status.results;
        }

        if (status.status === 'error') {
            throw new Error(status.error || 'Video processing failed on server.');
        }
    }
}

/* ── Show real results from API ─────────────────── */
let _galleryImages = [];  // batch images array
let _galleryIdx = 0;      // current gallery index
let _galleryIsVideo = false;

function showResults(data) {
    const resultsEl = document.getElementById('results-content');
    resultsEl.style.display = 'flex';
    window.latestInspectionData = data;

    const galleryNav = document.getElementById('gallery-nav');
    const isBatch = data.annotated_images && data.annotated_images.length > 1;

    if (isBatch) {
        // Batch mode (images) or video frames — set up gallery
        _galleryImages = data.annotated_images;
        _galleryIdx = 0;
        galleryNav.style.display = 'block';
        // Show timestamp label for video frames
        const isVideo = !!data.is_video;
        _galleryIsVideo = isVideo;
        showGalleryImage(0);
    } else {
        // Single image mode
        _galleryImages = [];
        galleryNav.style.display = 'none';
        if (data.annotated_image) {
            drawAnnotatedImage(data.annotated_image, data.detections);
        } else {
            drawBoxesOverUpload(data.detections);
        }
    }

    // Render per-object detection list
    renderDetectionList(data.detections);

    // Update 3D Model Batch Detections
    render3DDetections(data);

    // Update confidence bar
    const avgConf = data.summary.avg_confidence;
    const avgPct = Math.round(avgConf * 100);
    setTimeout(() => {
        const bar = document.getElementById('conf-bar');
        const val = document.getElementById('conf-value');
        if (bar) bar.style.width = avgPct + '%';
        if (val) val.textContent = avgPct + '%';
    }, 300);

    updateSummaryStats(data);
    populateAndShowReport(data);
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── 3D View Tab Switching & Rendering ── */
window.switchViewTab = function (tabId) {
    const btn2d = document.getElementById('tab-2d');
    const btn3d = document.getElementById('tab-3d');
    const view2d = document.getElementById('view-2d');
    const view3d = document.getElementById('view-3d');

    if (tabId === '2d') {
        btn2d.classList.add('active');
        btn2d.style.fontWeight = 'bold';
        btn2d.style.color = 'var(--teal)';
        btn2d.style.background = 'var(--bg-2)';
        btn2d.style.border = '1px solid var(--teal)';

        btn3d.classList.remove('active');
        btn3d.style.fontWeight = 'normal';
        btn3d.style.color = 'var(--text-3)';
        btn3d.style.background = 'var(--bg-panel)';
        btn3d.style.border = '1px solid var(--border-color)';

        view2d.style.display = 'block';
        view3d.style.display = 'none';
    } else {
        btn3d.classList.add('active');
        btn3d.style.fontWeight = 'bold';
        btn3d.style.color = 'var(--teal)';
        btn3d.style.background = 'var(--bg-2)';
        btn3d.style.border = '1px solid var(--teal)';

        btn2d.classList.remove('active');
        btn2d.style.fontWeight = 'normal';
        btn2d.style.color = 'var(--text-3)';
        btn2d.style.background = 'var(--bg-panel)';
        btn2d.style.border = '1px solid var(--border-color)';

        view3d.style.display = 'block';
        view2d.style.display = 'none';
    }
};

function render3DDetections(data) {
    const blipsContainer = document.getElementById('threed-blips');
    const countLabel = document.getElementById('threed-batch-count');
    if (!blipsContainer || !countLabel) return;

    blipsContainer.innerHTML = '';

    const isBatch = data.annotated_images && data.annotated_images.length > 1;
    let allDetections = [];
    if (isBatch) {
        data.annotated_images.forEach(imgData => {
            if (imgData.detections) allDetections.push(...imgData.detections);
        });
    } else {
        allDetections = data.detections || [];
    }
    countLabel.textContent = allDetections.length;

    const hullBounds = { minX: 26, maxX: 74, minY: 11, maxY: 49 };
    const SVG_NS = 'http://www.w3.org/2000/svg';

    const getSymbol = (className, cx, cy, color, conf, det) => {
        const key = (className || '').toLowerCase();
        const s = 3.8;
        const g = document.createElementNS(SVG_NS, 'g');
        g.setAttribute('transform', `translate(${cx},${cy})`);

        // Pulse ring
        const ring = document.createElementNS(SVG_NS, 'circle');
        ring.setAttribute('cx', 0); ring.setAttribute('cy', 0);
        ring.setAttribute('r', s + 1.5);
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', '0.5');
        ring.setAttribute('opacity', '0.35');
        if (conf > 0.7) {
            const anim = document.createElementNS(SVG_NS, 'animate');
            anim.setAttribute('attributeName', 'r');
            anim.setAttribute('values', `${s};${s+3};${s}`);
            anim.setAttribute('dur', conf > 0.85 ? '1.2s' : '2s');
            anim.setAttribute('repeatCount', 'indefinite');
            ring.appendChild(anim);
        }
        g.appendChild(ring);

        let shape;

        if (key.includes('paint_peel') || key.includes('paint')) {
            // Lightning bolt = paint peel danger
            shape = document.createElementNS(SVG_NS, 'polygon');
            shape.setAttribute('points', `0,${-s} ${-s*0.4},${-s*0.1} ${-s*0.1},${-s*0.1} ${-s*0.5},${s} ${s*0.4},${s*0.1} ${s*0.1},${s*0.1}`);
            shape.setAttribute('fill', color);
            shape.setAttribute('opacity', '0.95');
        } else if (key.includes('biofouling') || key.includes('fouling')) {
            // Leaf drop = biofouling
            shape = document.createElementNS(SVG_NS, 'path');
            shape.setAttribute('d', `M 0,${s} Q ${s*1.1},${s*0.2} 0,${-s} Q ${-s*1.1},${s*0.2} 0,${s}`);
            shape.setAttribute('fill', color);
            shape.setAttribute('opacity', '0.9');
            const stem = document.createElementNS(SVG_NS, 'line');
            stem.setAttribute('x1', 0); stem.setAttribute('y1', s);
            stem.setAttribute('x2', 0); stem.setAttribute('y2', s + 2);
            stem.setAttribute('stroke', color); stem.setAttribute('stroke-width', '0.7');
            g.appendChild(stem);
        } else if (key.includes('anode')) {
            // Circle = anode marker
            shape = document.createElementNS(SVG_NS, 'circle');
            shape.setAttribute('cx', '0');
            shape.setAttribute('cy', '0');
            shape.setAttribute('r', s);
            shape.setAttribute('fill', 'none');
            shape.setAttribute('stroke', color);
            shape.setAttribute('stroke-width', '1.5');
        } else if (key.includes('hull') || key.includes('corrosion')) {
            // Checkmark = corrosion marker
            shape = document.createElementNS(SVG_NS, 'polyline');
            shape.setAttribute('points', `${-s},0 ${-s*0.2},${s*0.7} ${s},${-s*0.7}`);
            shape.setAttribute('fill', 'none');
            shape.setAttribute('stroke', color);
            shape.setAttribute('stroke-width', '1.5');
            shape.setAttribute('stroke-linecap', 'round');
            shape.setAttribute('stroke-linejoin', 'round');
        } else {
            // Default diamond
            shape = document.createElementNS(SVG_NS, 'polygon');
            shape.setAttribute('points', `0,${-s} ${s},0 0,${s} ${-s},0`);
            shape.setAttribute('fill', color);
            shape.setAttribute('opacity', '0.85');
        }
        g.appendChild(shape);

        // Confidence % label below symbol
        const label = document.createElementNS(SVG_NS, 'text');
        label.setAttribute('x', 0); label.setAttribute('y', s + 4.5);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '2.4');
        label.setAttribute('fill', color);
        label.setAttribute('font-family', 'monospace');
        label.setAttribute('opacity', '0.9');
        label.textContent = Math.round(conf * 100) + '%';
        g.appendChild(label);

        // Species label if present
        if (det && det.species) {
            const sl = document.createElementNS(SVG_NS, 'text');
            sl.setAttribute('x', 0); sl.setAttribute('y', s + 7.5);
            sl.setAttribute('text-anchor', 'middle');
            sl.setAttribute('font-size', '1.8');
            sl.setAttribute('fill', '#fff');
            sl.setAttribute('font-family', 'monospace');
            sl.setAttribute('opacity', '0.75');
            sl.textContent = det.species.toUpperCase();
            g.appendChild(sl);
        }

        return g;
    };

    allDetections.forEach((det, i) => {
        const rx = hullBounds.minX + (Math.abs(Math.sin(i * 12.3 + 1.7)) * (hullBounds.maxX - hullBounds.minX));
        const ry = hullBounds.minY + (Math.abs(Math.cos(i * 7.4 + 0.9)) * (hullBounds.maxY - hullBounds.minY));
        const color = classColor(det.class_name);
        const group = getSymbol(normalizeClassName(det.class_name), rx, ry, color, det.confidence, det);
        blipsContainer.appendChild(group);
    });
}

/* ── Gallery navigation for batch images ── */
function showGalleryImage(idx) {
    if (!_galleryImages.length) return;
    _galleryIdx = Math.max(0, Math.min(idx, _galleryImages.length - 1));
    const item = _galleryImages[_galleryIdx];

    if (item.image) {
        drawAnnotatedImage(item.image, item.detections || []);
    }

    const counter  = document.getElementById('gallery-counter');
    const filename = document.getElementById('gallery-filename');
    if (counter) counter.textContent = `${_galleryIdx + 1} / ${_galleryImages.length}`;
    if (filename) {
        // Show timestamp for video frames, filename for batch images
        filename.textContent = item.timestamp_label
            ? `⏱ ${item.timestamp_label}  —  ${item.detections ? item.detections.length : 0} detection(s)`
            : (item.filename || '');
    }

    const prevBtn = document.getElementById('gallery-prev');
    const nextBtn = document.getElementById('gallery-next');
    if (prevBtn) prevBtn.disabled = _galleryIdx === 0;
    if (nextBtn) nextBtn.disabled = _galleryIdx === _galleryImages.length - 1;
}

window.galleryNav = function (dir) {
    showGalleryImage(_galleryIdx + dir);
};

/* ── Draw the base64 annotated image on canvas ── */
function drawAnnotatedImage(b64DataUri, detections) {
    const canvas = document.getElementById('result-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        // Resize canvas to image aspect
        const maxW = canvas.parentElement ? canvas.parentElement.offsetWidth || 640 : 640;
        const scale = Math.min(maxW / img.width, 1);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Because OpenCV already drew boxes on the image,
        // we optionally re-draw a HUD overlay border
        ctx.strokeStyle = 'rgba(0,200,176,0.25)';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, canvas.width, canvas.height);
    };

    img.src = b64DataUri;
}

/* ── Draw boxes on canvas over uploaded image ─── */
function drawBoxesOverUpload(detections) {
    const canvas = document.getElementById('result-canvas');
    const ctx = canvas.getContext('2d');
    const previewImg = document.getElementById('preview-image');

    const draw = (img) => {
        const W = canvas.width, H = canvas.height;

        // Scale factor (the preview image may have different natural dimensions)
        const scaleX = W / (img.naturalWidth || img.width || W);
        const scaleY = H / (img.naturalHeight || img.height || H);

        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(img, 0, 0, W, H);

        detections.forEach(det => {
            const x1 = det.x1 * scaleX, y1 = det.y1 * scaleY;
            const x2 = det.x2 * scaleX, y2 = det.y2 * scaleY;
            const w = x2 - x1, h = y2 - y1;
            const color = classColor(det.class_name);
            const confPct = Math.round(det.confidence * 100) + '%';

            // Box glow
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, w, h);
            ctx.shadowBlur = 0;

            // Corner marks
            const cs = 10;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            [[x1, y1, cs, cs], [x2, y1, -cs, cs], [x1, y2, cs, -cs], [x2, y2, -cs, -cs]].forEach(([cx, cy, dx, dy]) => {
                ctx.beginPath(); ctx.moveTo(cx + dx, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy); ctx.stroke();
            });

            // Label
            const label = `${normalizeClassName(det.class_name)}  ${confPct}`;
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            const tw = ctx.measureText(label).width + 14;
            const lh = 18;
            const ly = Math.max(y1 - lh - 2, 2);
            ctx.fillStyle = color;
            ctx.fillRect(x1, ly, tw, lh + 4);
            ctx.fillStyle = '#0a1520';
            ctx.fillText(label, x1 + 7, ly + lh - 1);
        });

        // Detection count
        const cLabel = `${detections.length} detection${detections.length !== 1 ? 's' : ''} found`;
        ctx.font = '600 11px "JetBrains Mono", monospace';
        const cw = ctx.measureText(cLabel).width + 16;
        ctx.fillStyle = 'rgba(5,10,15,0.8)';
        ctx.fillRect(W - cw - 4, H - 26, cw, 22);
        ctx.fillStyle = '#00c8b0';
        ctx.fillText(cLabel, W - cw, H - 9);
    };

    // If image already loaded
    if (previewImg && previewImg.naturalWidth) {
        draw(previewImg);
    } else if (previewImg) {
        previewImg.addEventListener('load', () => draw(previewImg), { once: true });
    }
}

/* ── Render per-object detection rows ──────────── */
function renderDetectionList(detections) {
    const list = document.getElementById('detections-list');
    list.innerHTML = '';

    if (!detections || detections.length === 0) {
        list.innerHTML = '<div style="color:var(--text-3);font-family:var(--font-mono);font-size:13px;padding:16px 0;">No objects detected</div>';
        return;
    }

    // Individual boxes with per-detection confidence
    const perBox = document.getElementById('per-detection-boxes');
    if (!perBox) return;
    perBox.innerHTML = '';

    detections.forEach(det => {
        const el = document.createElement('div');
        el.className = 'det-item';
        const color = classColor(det.class_name);
        const pct = Math.round(det.confidence * 100) + '%';

        let speciesHtml = '';
        if (det.species) {
            const speciesPct = Math.round(det.species_confidence * 100) + '%';
            speciesHtml = `
            <div style="font-size:10px; color:var(--text-2); margin-top:2px; font-family:var(--font-mono);">
              ↳ ResNet Species: <span style="color:#fff; font-weight:bold;">${det.species.toUpperCase()}</span> (${speciesPct})
            </div>`;
        }

        el.innerHTML = `
            <div>
              <div style="display:flex;align-items:center;gap:6px;">
                <span class="det-dot" style="background:${color};box-shadow:0 0 8px ${color}"></span>
                <span style="font-weight:600;font-size:13px;">${normalizeClassName(det.class_name)}</span>
              </div>
              <div style="font-size:11px;color:var(--text-3);margin-top:4px;">Pipeline Confidence: ${pct}</div>
              ${speciesHtml}
            </div>
            <div style="text-align:right;">
               <div style="font-size:11px;color:var(--text-3);font-family:var(--font-mono);">Bounding Box Area</div>
               <div style="font-size:12px;color:#fff;">${Math.round(det.x2 - det.x1)}x${Math.round(det.y2 - det.y1)}</div>
            </div>
        `;
        list.appendChild(el);
    });
}

/* ── Update summary HUD numbers on results panel ─ */
function updateSummaryStats(data) {
    const s = data.summary;
    const $id = (id) => document.getElementById(id);

    if ($id('result-total')) $id('result-total').textContent = s.total;
    if ($id('result-risk')) $id('result-risk').textContent = s.risk_level;
    if ($id('result-inf-ms')) $id('result-inf-ms').textContent = s.inference_time_ms + ' ms';
    if ($id('result-insp-id')) $id('result-insp-id').textContent = data.inspection_id;

    // Colour the risk badge
    const riskEl = $id('result-risk');
    if (riskEl) {
        const rColors = { HIGH: '#e74c3c', MEDIUM: '#f0a500', LOW: '#00c8b0', SAFE: '#00c8b0' };
        riskEl.style.color = rColors[s.risk_level] || 'var(--text-1)';
    }
}

/* ── Reveal report section with real data ────────── */
function populateAndShowReport(data) {
    const reportSection = document.getElementById('report');
    if (!reportSection) return;

    // Show the report section
    reportSection.style.display = '';
    reportSection.removeAttribute('hidden');

    const s = data.summary;
    const mm = data.model_metrics;

    // Patch report card fields
    const setEl = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    setEl('rpt-inspection-id', data.inspection_id);
    setEl('rpt-date', new Date(data.timestamp).toLocaleString());
    setEl('rpt-risk', s.risk_level);
    setEl('rpt-total', s.total + ' anomalies detected');
    setEl('rpt-inf-time', s.inference_time_ms + ' ms');
    setEl('rpt-precision', (mm.precision * 100).toFixed(1) + '%');
    setEl('rpt-recall', (mm.recall * 100).toFixed(1) + '%');
    setEl('rpt-map50', (mm.map50 * 100).toFixed(1) + '%');

    // Risk banner colour
    const riskBanner = document.getElementById('rpt-risk-banner');
    if (riskBanner) {
        riskBanner.className = riskBanner.className.replace(/rpt-risk-\w+/g, '');
        const cls = { HIGH: 'rpt-risk-high', MEDIUM: 'rpt-risk-moderate', LOW: 'rpt-risk-low', SAFE: 'rpt-risk-low' };
        riskBanner.classList.add(cls[s.risk_level] || 'rpt-risk-low');
    }

    updateReportSummaryCounts(data.detections || []);
    updateReportAnnotatedImage(data);

    // Detection breakdown in report
    const breakdown = document.getElementById('rpt-detection-breakdown');
    if (breakdown && data.detections) {
        const normalizeReportClass = (name) => {
            const key = (name || '').toLowerCase().trim();
            if (!key) return key;
            if (key.includes('biofouling') || key.includes('fouling')) return 'biofouling';
            if (key.includes('paint_peel') || key.includes('paint')) return 'paint_peel';
            if (key.includes('anode')) return 'anode';
            if (key.includes('hull') || key.includes('corrosion')) return 'corrosion';
            return key;
        };
        const groups = {};
        data.detections.forEach(d => {
            const key = normalizeReportClass(d.class_name);
            if (!groups[key]) groups[key] = { total: 0, maxConf: 0 };
            groups[key].total++;
            groups[key].maxConf = Math.max(groups[key].maxConf, d.confidence);
        });

        breakdown.innerHTML = Object.entries(groups).map(([cls, g]) => {
            const pct = Math.round(g.maxConf * 100);
            const color = classColor(cls);
            return `
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
            <span style="color:var(--text-2);">${cls}</span>
            <span style="font-family:var(--font-mono);color:${color};">${pct}%</span>
          </div>
          <div style="height:4px;background:var(--border-dim);border-radius:2px;">
            <div style="width:${pct}%;height:100%;background:${color};border-radius:2px;transition:width 0.8s;"></div>
          </div>
        </div>`;
        }).join('');
    }

    // Scroll to report
    setTimeout(() => {
        reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 600);
}

function updateReportSummaryCounts(detections) {
    const grid = document.getElementById('rpt-summary-grid');
    if (!grid) return;

    // Always show all 4 classes, even if count is 0
    const CLASSES = [
        { key: 'corrosion',  label: 'Corrosion',  color: '#e74c3c' },
        { key: 'biofouling', label: 'Biofouling', color: '#f0a500' },
        { key: 'anode',      label: 'Anode',      color: '#00d4e8' },
        { key: 'paint_peel', label: 'Paint Peel', color: '#e74c3c' },
    ];

    const counts = { corrosion: 0, biofouling: 0, anode: 0, paint_peel: 0 };

    (detections || []).forEach(d => {
        const k = (d.class_name || '').toLowerCase().trim();
        if (k === 'hull' || k === 'corrosion')                        counts.corrosion++;
        else if (k === 'biofouling' || k.includes('fouling'))         counts.biofouling++;
        else if (k === 'anode')                                        counts.anode++;
        else if (k === 'paint_peel' || k.includes('paint'))           counts.paint_peel++;
    });

    grid.innerHTML = CLASSES.map(c => `
        <div class="rpt-summary-item">
            <div class="rsi-val" style="color:${c.color};">${counts[c.key]}</div>
            <div class="rsi-label">${c.label}</div>
        </div>`).join('');
}

function updateReportAnnotatedImage(data) {
    const imgEl = document.getElementById('rpt-annotated-img');
    const layer = document.getElementById('rpt-bbox-layer');
    const empty = document.getElementById('rpt-annotated-empty');
    const rptGallery = document.getElementById('rpt-gallery');
    const rptGrid = document.getElementById('rpt-gallery-grid');
    if (!imgEl || !layer) return;

    const isBatch = data.annotated_images && data.annotated_images.length > 1;

    if (isBatch && rptGallery && rptGrid) {
        // Show first image as main
        const first = data.annotated_images[0];
        if (first && first.image) {
            imgEl.src = first.image;
            imgEl.style.display = 'block';
            if (empty) empty.style.display = 'none';
        }
        layer.innerHTML = '';

        // Build thumbnail gallery grid
        rptGallery.style.display = 'block';
        rptGrid.innerHTML = data.annotated_images.map((item, i) => {
            const src = item.image || '';
            const detCount = item.detections ? item.detections.length : 0;
            if (!src) return '';
            return `<div style="cursor:pointer;border:1px solid var(--border-dim);border-radius:4px;overflow:hidden;background:var(--bg-2);transition:border-color 0.2s;"
                        onmouseover="this.style.borderColor='var(--teal)'" onmouseout="this.style.borderColor='var(--border-dim)'"
                        onclick="document.getElementById('rpt-annotated-img').src='${src}';">
                <img src="${src}" alt="${item.filename}" style="width:100%;height:100px;object-fit:cover;display:block;" />
                <div style="padding:4px 6px;font-size:10px;font-family:var(--font-mono);">
                    <div style="color:var(--text-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${item.filename}">${item.filename}</div>
                    <div style="color:var(--teal);">${detCount} detection${detCount !== 1 ? 's' : ''}</div>
                </div>
            </div>`;
        }).join('');
        return;
    }

    // Single image mode (original logic)
    if (rptGallery) rptGallery.style.display = 'none';
    const annotatedSrc = data.annotated_image || '';
    const previewImg = document.getElementById('preview-image');
    const fallbackSrc = previewImg && previewImg.src ? previewImg.src : '';
    const imgSrc = annotatedSrc || fallbackSrc;

    layer.innerHTML = '';

    if (!imgSrc) {
        imgEl.style.display = 'none';
        if (empty) empty.style.display = 'flex';
        return;
    }

    imgEl.src = imgSrc;
    imgEl.style.display = 'block';
    if (empty) empty.style.display = 'none';

    if (annotatedSrc || !data.detections || data.detections.length === 0) {
        return;
    }

    const draw = () => {
        layer.innerHTML = '';
        const iw = imgEl.naturalWidth || imgEl.width;
        const ih = imgEl.naturalHeight || imgEl.height;
        if (!iw || !ih) return;

        data.detections.forEach(det => {
            const x1 = (det.x1 / iw) * 100;
            const y1 = (det.y1 / ih) * 100;
            const x2 = (det.x2 / iw) * 100;
            const y2 = (det.y2 / ih) * 100;
            const w = x2 - x1;
            const h = y2 - y1;
            const color = classColor(det.class_name);
            const confPct = Math.round(det.confidence * 100) + '%';

            const box = document.createElement('div');
            box.className = 'rpt-bbox';
            box.style.cssText = `top:${y1}%;left:${x1}%;width:${w}%;height:${h}%;border-color:${color};`;

            const label = document.createElement('div');
            label.className = 'rpt-bbox-label';
            label.style.cssText = `background:${color};color:#0a1520;`;
            label.textContent = `${normalizeClassName(det.class_name)} ${confPct}`;
            box.appendChild(label);

            layer.appendChild(box);
        });
    };

    if (imgEl.complete) draw();
    else imgEl.addEventListener('load', draw, { once: true });
}

/* ── Demo fallback (when API is offline) ────────── */
const DEMO_DETECTIONS = [
    { class_name: 'biofouling',  confidence: 0.91, x1: 58,  y1: 62,  x2: 198, y2: 222 },
    { class_name: 'paint_peel', confidence: 0.88, x1: 328, y1: 305, x2: 448, y2: 445 },
    { class_name: 'biofouling', confidence: 0.85, x1: 554, y1: 155, x2: 668, y2: 277 },
    { class_name: 'anode',      confidence: 0.94, x1: 420, y1: 40,  x2: 524, y2: 152 },
    { class_name: 'corrosion',  confidence: 0.99, x1: 618, y1: 318, x2: 778, y2: 458 },
    { class_name: 'paint_peel', confidence: 0.81, x1: 110, y1: 380, x2: 244, y2: 502 },
];

function showDemoResults() {
    const emptyState = document.getElementById('empty-state');
    const resultsEl = document.getElementById('results-content');
    emptyState.style.display = 'none';
    resultsEl.style.display = 'flex';

    const demoData = {
        inspection_id: 'DEMO-0000',
        detections: DEMO_DETECTIONS,
        annotated_image: null,
        summary: {
            total: DEMO_DETECTIONS.length,
            risk_level: 'MEDIUM',
            avg_confidence: 0.897,
            max_confidence: 0.99,
            inference_time_ms: 42,
        },
        model_metrics: { precision: 0.886, recall: 0.844, map50: 0.882, map5095: 0.782 },
        timestamp: new Date().toISOString(),
    };
    window.latestInspectionData = demoData;

    drawBoxesOverUpload(DEMO_DETECTIONS);
    renderDetectionList(DEMO_DETECTIONS);

    const bar = document.getElementById('conf-bar');
    const val = document.getElementById('conf-value');
    if (bar) bar.style.width = '90%';
    if (val) val.textContent = '90%';

    updateSummaryStats(demoData);
    populateAndShowReport(demoData);

    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Navigate to report section ─────────────────── */
window.generateReport = function () {
    const reportSection = document.getElementById('report');
    if (reportSection) {
        reportSection.scrollIntoView({ behavior: 'smooth' });
        const demoCard = document.getElementById('report-demo');
        if (demoCard) {
            demoCard.style.borderColor = 'var(--teal)';
            demoCard.style.boxShadow = '0 0 40px rgba(0,200,176,0.3)';
            setTimeout(() => {
                demoCard.style.borderColor = '';
                demoCard.style.boxShadow = '';
            }, 2000);
        }
    }
};

/* ── Utilities ──────────────────────────────────── */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showToast(msg, type = 'info') {
    const colors = { info: '#00c8b0', warn: '#f0a500', error: '#e74c3c' };
    const toast = document.createElement('div');
    toast.style.cssText = `
    position:fixed;bottom:28px;right:28px;z-index:9999;
    background:var(--bg-panel);border:1px solid ${colors[type]};
    color:var(--text-1);padding:12px 20px;border-radius:4px;
    font-family:var(--font-mono);font-size:13px;
    box-shadow:0 4px 24px rgba(0,0,0,0.6);max-width:380px;line-height:1.5;
  `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

/* ── 3D Interaction Logic ── */
let threedInteracted = false;
let threedRotZ = 0;
let threedIsDragging = false;
let threedStartX = 0;

function init3DInteraction() {
    const svg = document.getElementById('threed-svg');
    if (!svg) return;

    function animate() {
        if (!threedInteracted) {
            threedRotZ += 0.5; // Clockwise rotation
            svg.style.transform = `rotateX(60deg) rotateZ(${threedRotZ}deg)`;
        }
        requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);

    svg.addEventListener('mousedown', (e) => {
        threedIsDragging = true;
        threedInteracted = true;
        threedStartX = e.clientX;
        svg.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!threedIsDragging) return;
        const deltaX = e.clientX - threedStartX;
        threedRotZ += deltaX * 0.5;
        svg.style.transform = `rotateX(60deg) rotateZ(${threedRotZ}deg)`;
        threedStartX = e.clientX;
    });

    window.addEventListener('mouseup', () => {
        if (!threedIsDragging) return;
        threedIsDragging = false;
        if (svg) svg.style.cursor = 'grab';
        // Resume auto-rotation after 2 seconds of inactivity
        setTimeout(() => { if (!threedIsDragging) threedInteracted = false; }, 2000);
    });

    // Also handle touch events for mobile
    svg.addEventListener('touchstart', (e) => {
        threedIsDragging = true;
        threedInteracted = true;
        threedStartX = e.touches[0].clientX;
    });
    window.addEventListener('touchmove', (e) => {
        if (!threedIsDragging) return;
        const deltaX = e.touches[0].clientX - threedStartX;
        threedRotZ += deltaX * 0.5;
        svg.style.transform = `rotateX(60deg) rotateZ(${threedRotZ}deg)`;
        threedStartX = e.touches[0].clientX;
    });
    window.addEventListener('touchend', () => {
        if (!threedIsDragging) return;
        threedIsDragging = false;
        setTimeout(() => { if (!threedIsDragging) threedInteracted = false; }, 2000);
    });
}
document.addEventListener('DOMContentLoaded', init3DInteraction);


/* ═══════════════════════════════════════════════════════════════════════════
   BULK / TB-SCALE UPLOAD  —  uses POST /batch-job + GET /job/{id}/status
   ═══════════════════════════════════════════════════════════════════════════ */

let _bulkPollTimer   = null;   // setInterval handle
let _activeBulkJobId = null;   // current job being polled

/* ── Inject the bulk progress panel into the page (once) ──────────────────── */
(function injectBulkPanel() {
    if (document.getElementById('bulk-panel')) return;

    const panel = document.createElement('div');
    panel.id = 'bulk-panel';
    panel.style.cssText = `
        display:none;
        position:fixed;bottom:80px;right:28px;z-index:9990;
        width:360px;background:var(--bg-panel,#0d1b2a);
        border:1px solid var(--teal,#00c8b0);border-radius:8px;
        padding:18px 20px;box-shadow:0 8px 40px rgba(0,0,0,0.7);
        font-family:var(--font-mono,'JetBrains Mono',monospace);font-size:12px;
        color:var(--text-1,#e0f0ff);
    `;
    panel.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:13px;font-weight:700;color:var(--teal,#00c8b0);">
                🚀 Bulk Job Processing
            </span>
            <button id="bulk-panel-close" onclick="closeBulkPanel()"
                style="background:none;border:none;color:var(--text-3,#7a9eb5);font-size:16px;cursor:pointer;line-height:1;">✕</button>
        </div>

        <!-- Job ID row -->
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:var(--text-3,#7a9eb5);">Job ID</span>
            <span id="bulk-job-id" style="color:var(--teal,#00c8b0);letter-spacing:0.5px;">—</span>
        </div>

        <!-- Status row -->
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:var(--text-3,#7a9eb5);">Status</span>
            <span id="bulk-status" style="color:#f0a500;text-transform:uppercase;">queued</span>
        </div>

        <!-- Files row -->
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:var(--text-3,#7a9eb5);">Files</span>
            <span id="bulk-files">0 / 0</span>
        </div>

        <!-- Current file -->
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:var(--text-3,#7a9eb5);">Current</span>
            <span id="bulk-current" style="color:var(--text-2,#aaccdd);max-width:200px;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right;">—</span>
        </div>

        <!-- Detections so far -->
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:var(--text-3,#7a9eb5);">Detections</span>
            <span id="bulk-dets" style="color:var(--teal,#00c8b0);">0</span>
        </div>

        <!-- Elapsed time -->
        <div style="display:flex;justify-content:space-between;margin-bottom:14px;">
            <span style="color:var(--text-3,#7a9eb5);">Elapsed</span>
            <span id="bulk-elapsed">0s</span>
        </div>

        <!-- Progress bar -->
        <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;margin-bottom:6px;">
            <div id="bulk-bar" style="height:6px;border-radius:4px;background:var(--teal,#00c8b0);
                width:0%;transition:width 0.5s ease;"></div>
        </div>
        <div style="text-align:right;color:var(--text-3,#7a9eb5);font-size:11px;">
            <span id="bulk-pct">0%</span>
        </div>

        <!-- ETA -->
        <div id="bulk-eta-row" style="margin-top:8px;color:var(--text-3,#7a9eb5);font-size:11px;text-align:center;"></div>

        <!-- Done message -->
        <div id="bulk-done-msg" style="display:none;margin-top:12px;text-align:center;
            padding:8px;background:rgba(0,200,176,0.1);border:1px solid var(--teal,#00c8b0);
            border-radius:4px;color:var(--teal,#00c8b0);font-size:12px;">
            ✅ Processing complete! Results loaded below.
        </div>
    `;
    document.body.appendChild(panel);
})();


/* ── Open/close the panel ─────────────────────────────────────────────────── */
function openBulkPanel()  { const p = document.getElementById('bulk-panel'); if (p) p.style.display = 'block'; }
window.closeBulkPanel = function() {
    const p = document.getElementById('bulk-panel');
    if (p) p.style.display = 'none';
    // Keep polling in background even if panel is hidden
};


/* ── Update panel fields from a job status object ───────────────────────────── */
function _updateBulkPanel(job) {
    const $ = id => document.getElementById(id);
    if (!$('bulk-panel')) return;

    const statusColors = { queued: '#f0a500', running: '#00c8b0', done: '#00c8b0', error: '#e74c3c' };
    $('bulk-job-id').textContent  = job.job_id || '—';
    $('bulk-status').textContent  = job.status || '—';
    $('bulk-status').style.color  = statusColors[job.status] || '#f0a500';
    $('bulk-files').textContent   = `${job.processed_files} / ${job.total_files}`;
    $('bulk-current').textContent = job.current_file || '—';
    $('bulk-dets').textContent    = job.detections_so_far || 0;
    $('bulk-elapsed').textContent = `${job.elapsed_sec || 0}s`;

    const pct = job.progress_pct || 0;
    $('bulk-bar').style.width = pct + '%';
    $('bulk-pct').textContent = pct + '%';

    // ETA estimate
    const etaRow = $('bulk-eta-row');
    if (job.status === 'running' && job.elapsed_sec > 2 && job.processed_files > 0) {
        const rate = job.processed_files / job.elapsed_sec;             // files/sec
        const remaining = (job.total_files - job.processed_files) / rate;
        etaRow.textContent = `ETA ≈ ${Math.ceil(remaining)}s  (${rate.toFixed(1)} files/s)`;
    } else {
        etaRow.textContent = '';
    }

    // Done state
    if (job.status === 'done') {
        $('bulk-done-msg').style.display = 'block';
    }
}


/* ── Poll job status every 2 seconds ─────────────────────────────────────── */
function _startPolling(jobId) {
    _activeBulkJobId = jobId;
    if (_bulkPollTimer) clearInterval(_bulkPollTimer);

    _bulkPollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`${API_BASE}/job/${jobId}/status`);
            if (!resp.ok) return;
            const job = await resp.json();

            _updateBulkPanel(job);

            if (job.status === 'done') {
                clearInterval(_bulkPollTimer);
                _bulkPollTimer = null;
                showToast(`✅ Batch job done! ${job.total_files} files, ${job.detections_so_far} detections.`, 'info');
                // Show results in main dashboard
                if (job.results) {
                    const emptyState   = document.getElementById('empty-state');
                    const loadingState = document.getElementById('loading-state');
                    if (emptyState)   emptyState.style.display = 'none';
                    if (loadingState) loadingState.style.display = 'none';
                    showResults(job.results);
                }
            } else if (job.status === 'error') {
                clearInterval(_bulkPollTimer);
                _bulkPollTimer = null;
                showToast(`❌ Batch job failed: ${job.error}`, 'error');
            }
        } catch (e) {
            // Network error — keep polling, API may be momentarily busy
        }
    }, 2000);
}


/* ── Submit batch job to backend ─────────────────────────────────────────── */
window.runBulkDetection = async function(inputFiles) {
    const files = inputFiles || uploadedFiles;
    if (!files || files.length === 0) {
        showToast('Please select files to upload first.', 'warn');
        return;
    }

    showToast(`Submitting ${files.length} files to batch queue…`, 'info');
    openBulkPanel();

    // Reset panel
    const doneMsg = document.getElementById('bulk-done-msg');
    if (doneMsg) doneMsg.style.display = 'none';

    try {
        const formData = new FormData();
        for (const f of files) {
            formData.append('files', f);
        }

        const resp = await fetch(`${API_BASE}/batch-job`, {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const job = await resp.json();
        console.log('[BULK] Job submitted:', job.job_id);

        // Initial panel state
        _updateBulkPanel({
            job_id:           job.job_id,
            status:           'queued',
            total_files:      job.total_files,
            processed_files:  0,
            current_file:     '',
            detections_so_far: 0,
            progress_pct:     0,
            elapsed_sec:      0,
        });

        _startPolling(job.job_id);

    } catch (e) {
        showToast(`Batch submission failed: ${e.message}`, 'error');
        if (document.getElementById('bulk-panel')) {
            document.getElementById('bulk-panel').style.display = 'none';
        }
    }
};


/* ── Bulk mode button handler ("Bulk (TB)" button in HTML) ───────────────── */
window.setUploadType = (function(_orig) {
    return function(type) {
        _orig(type);
        if (type === 'bulk') {
            // Automatically run bulk detection after file selection completes
            const waitForFiles = setInterval(() => {
                if (uploadedFiles.length > 0) {
                    clearInterval(waitForFiles);
                    window.runBulkDetection(uploadedFiles);
                }
            }, 500);
            // Clear wait after 30s to avoid leak
            setTimeout(() => clearInterval(waitForFiles), 30000);
        }
    };
})(window.setUploadType);

