/* ═══════════════════════════════════════════════
   NautiCAI — Report JavaScript
   Fully dynamic — all values from real detection data
═══════════════════════════════════════════════ */

/* ── Color map for any class name ─────────────── */
function reportColor(name) {
  const k = (name || '').toLowerCase().trim();
  if (k === 'corrosion' || k === 'hull' || k.includes('paint'))  return '#e74c3c';
  if (k === 'biofouling' || k === 'marine_growth' || k.includes('fouling')) return '#f0a500';
  if (k === 'anode')    return '#00d4e8';
  if (k === 'debris')   return '#27ae60';
  if (k === 'anomaly')  return '#9b59b6';
  return '#00b4a0';
}

/* ── Normalize class name for display ─────────── */
function reportLabel(name) {
  return (name || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/* ── Download Report ──────────────────────────── */
window.downloadReport = function () {
  const btn = event.currentTarget || document.querySelector('.report-actions .btn-primary');
  const originalText = btn ? btn.innerHTML : '';
  const data = window.latestInspectionData || null;

  if (btn) {
    btn.innerHTML = `<svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg> Generating PDF...`;
    btn.disabled = true;
    btn.style.opacity = '0.7';
  }

  setTimeout(() => {
    const blob = new Blob([generateReportHTML(data)], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = data?.inspection_id
      ? `NautiCAI_Report_${data.inspection_id}.html`
      : 'NautiCAI_Report.html';
    a.click();
    URL.revokeObjectURL(url);
    if (btn) { btn.innerHTML = originalText; btn.disabled = false; btn.style.opacity = ''; }
  }, 1200);
};

/* ── Generate fully-dynamic report HTML ────────── */
function generateReportHTML(data) {
  const detections = Array.isArray(data?.detections) ? data.detections : [];
  const reportId   = data?.inspection_id || '—';
  const timestamp  = data?.timestamp ? new Date(data.timestamp) : new Date();
  const dateStr    = timestamp.toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' });
  const riskLevel  = data?.summary?.risk_level || 'SAFE';
  const infTime    = data?.summary?.inference_time_ms != null ? `${data.summary.inference_time_ms} ms` : '—';
  const mm         = data?.model_metrics || {};

  // All defect classes across all 3 models — healthy_surface suppressed
  const CLASSES = [
    { key: 'corrosion',  label: 'Corrosion',  color: '#e74c3c' },
    { key: 'biofouling', label: 'Biofouling', color: '#f0a500' },
    { key: 'debris',     label: 'Debris',     color: '#27ae60' },
    { key: 'anomaly',    label: 'Anomaly',    color: '#9b59b6' },
    { key: 'anode',      label: 'Anode',      color: '#00d4e8' },
    { key: 'paint_peel', label: 'Paint Peel', color: '#e74c3c' },
  ];
  const counts = { corrosion: 0, biofouling: 0, debris: 0, anomaly: 0, anode: 0, paint_peel: 0 };
  detections.forEach(d => {
    const k = (d.class_name || '').toLowerCase().trim();
    if (k === 'hull' || k === 'corrosion')                                    counts.corrosion++;
    else if (k === 'biofouling' || k === 'marine_growth' || k.includes('fouling')) counts.biofouling++;
    else if (k === 'debris')                                                  counts.debris++;
    else if (k === 'anomaly')                                                 counts.anomaly++;
    else if (k === 'anode')                                                   counts.anode++;
    else if (k === 'paint_peel' || k.includes('paint'))                       counts.paint_peel++;
  });

  // Summary stat cards — all 4 always visible
  const statCards = CLASSES.map(c => `
    <div style="text-align:center;padding:16px;border:1px solid #e0e0e0;border-radius:8px;">
      <div style="font-size:28px;font-weight:700;color:${c.color};">${counts[c.key]}</div>
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">${c.label}</div>
    </div>`).join('');

  // Confidence table — all 4 rows always shown
  const confRows = CLASSES.map(c => {
    const cnt = counts[c.key];
    const maxConf = detections
      .filter(d => {
        const k = (d.class_name || '').toLowerCase().trim();
        if (c.key === 'corrosion')  return k === 'hull' || k === 'corrosion';
        if (c.key === 'biofouling') return k === 'biofouling' || k.includes('fouling');
        if (c.key === 'paint_peel') return k === 'paint_peel' || k.includes('paint');
        return k === c.key;
      })
      .reduce((m, d) => Math.max(m, d.confidence || 0), 0);
    const pct = Math.round(maxConf * 100);
    return `<tr>
      <td style="padding:8px 12px;">${c.label}</td>
      <td style="padding:8px 12px;">${cnt}</td>
      <td style="padding:8px 12px;width:160px;">
        <div style="background:#eee;border-radius:4px;height:8px;">
          <div style="width:${pct}%;background:${c.color};height:8px;border-radius:4px;"></div>
        </div>
      </td>
      <td style="padding:8px 12px;color:${c.color};font-weight:600;">${pct > 0 ? pct + '%' : '—'}</td>
    </tr>`;
  }).join('');

  // Recommendations — dynamic based on what was found
  const recs = [];
  if (counts.biofouling > 0)
    recs.push({ level: 'HIGH', color: '#fde8e8', text: 'Schedule hull cleaning and anti-fouling treatment within 30 days.' });
  if (counts.paint_peel > 0)
    recs.push({ level: 'HIGH', color: '#fde8e8', text: 'Apply anti-corrosion primer and repaint affected sections within 30 days.' });
  if (counts.anode > 0)
    recs.push({ level: 'MEDIUM', color: '#e6f9fb', text: 'Inspect anode depletion and replace sacrificial anodes if >50% depleted.' });
  if (counts.corrosion > 0)
    recs.push({ level: 'MEDIUM', color: '#fff8e1', text: 'Monitor corrosion zones. Schedule structural assessment within 60 days.' });
  if (recs.length === 0)
    recs.push({ level: 'LOW', color: '#e8f5e9', text: 'No significant anomalies detected. Continue routine inspection schedule.' });

  const recsHtml = recs.map(r => `
    <div style="display:flex;gap:10px;align-items:flex-start;padding:10px 14px;border:1px solid #eee;border-radius:6px;margin-bottom:8px;font-size:13px;color:#444;">
      <span style="padding:2px 8px;background:${r.color};color:#333;border:1px solid #ccc;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap;">${r.level}</span>
      ${r.text}
    </div>`).join('');

  // Image pages — one per frame/image
  const isBatch = Array.isArray(data?.annotated_images) && data.annotated_images.length > 1;
  const imagePages = isBatch
    ? data.annotated_images.map((item, i) => buildImagePage(item.image, item.filename || item.timestamp_label, item.detections, i + 1)).join('')
    : buildImagePage(data?.annotated_image, '', data?.detections, 1);

  const riskColor = { HIGH: '#e74c3c', MEDIUM: '#f39c12', LOW: '#00b4a0', SAFE: '#00b4a0' };
  const rColor = riskColor[riskLevel] || '#00b4a0';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NautiCAI Inspection Report — ${reportId}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',sans-serif;background:#fff;color:#1a1a2e;}
  table{border-collapse:collapse;width:100%;}
  td,th{border:1px solid #e0e0e0;font-size:13px;}
  th{background:#f8fafb;font-weight:600;padding:8px 12px;text-align:left;}
  @media print{.page{padding:20px 28px!important;}}
</style>
</head>
<body>
<div style="max-width:800px;margin:0 auto;padding:40px 48px;">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:20px;border-bottom:3px solid #00b4a0;margin-bottom:32px;">
    <div>
      <div style="font-size:22px;font-weight:700;color:#0b1e2d;">NautiCAI</div>
      <div style="font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;">Underwater Inspection Platform</div>
    </div>
    <div style="text-align:right;font-size:12px;color:#888;">
      <div style="font-weight:600;color:#0b1e2d;">${reportId}</div>
      <div>${dateStr}</div>
      <div style="margin-top:4px;padding:3px 10px;background:${rColor}22;color:${rColor};border:1px solid ${rColor}55;border-radius:4px;font-weight:700;font-size:11px;display:inline-block;">${riskLevel} RISK</div>
    </div>
  </div>

  <!-- Title -->
  <div style="margin-bottom:28px;">
    <div style="font-size:26px;font-weight:700;color:#0b1e2d;margin-bottom:6px;">Subsea Hull Inspection Report</div>
    <div style="font-size:13px;color:#888;">AI-powered analysis · YOLOv8 + ResNet-50 Species Classifier</div>
  </div>

  <!-- Summary metadata -->
  <table style="margin-bottom:28px;">
    <tr><th>Inspection ID</th><td>${reportId}</td><th>Date</th><td>${dateStr}</td></tr>
    <tr><th>Risk Level</th><td style="color:${rColor};font-weight:600;">${riskLevel}</td><th>Inference Time</th><td>${infTime}</td></tr>
    <tr><th>Total Detections</th><td>${detections.length}</td><th>Classes Found</th><td>${Object.keys(counts).map(reportLabel).join(', ') || 'None'}</td></tr>
    <tr><th>Precision</th><td>${mm.precision != null ? (mm.precision*100).toFixed(1)+'%' : '—'}</td><th>Recall</th><td>${mm.recall != null ? (mm.recall*100).toFixed(1)+'%' : '—'}</td></tr>
    <tr><th>mAP@0.5</th><td>${mm.map50 != null ? (mm.map50*100).toFixed(1)+'%' : '—'}</td><th>mAP@0.5:0.95</th><td>${mm.map5095 != null ? (mm.map5095*100).toFixed(1)+'%' : '—'}</td></tr>
  </table>

  <!-- Detection stat cards -->
  <div style="font-size:12px;font-weight:600;color:#0b1e2d;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Detection Summary</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-bottom:28px;">
    ${statCards}
  </div>

  <!-- Confidence breakdown table -->
  <div style="font-size:12px;font-weight:600;color:#0b1e2d;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Confidence Breakdown</div>
  <table style="margin-bottom:28px;">
    <thead><tr><th>Class</th><th>Count</th><th>Confidence Bar</th><th>Max Conf</th></tr></thead>
    <tbody>${confRows}</tbody>
  </table>

  <!-- Recommendations -->
  <div style="font-size:12px;font-weight:600;color:#0b1e2d;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #eee;padding-bottom:6px;margin-bottom:14px;">Recommendations</div>
  <div style="margin-bottom:28px;">${recsHtml}</div>

</div>

${imagePages}

</body>
</html>`;
}

function buildImagePage(imgSrc, label, dets, pageNum) {
  const counts = {};
  (dets || []).forEach(d => {
    let n = (d.class_name || '').trim();
    if (n.toLowerCase() === 'hull') n = 'corrosion';
    if (n) counts[n] = (counts[n] || 0) + 1;
  });

  const tags = Object.entries(counts).map(([name, cnt]) => {
    const color = reportColor(name);
    return `<span style="display:inline-block;margin:3px 4px;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:600;background:${color}22;color:${color};border:1px solid ${color}55;">${reportLabel(name)}: ${cnt}</span>`;
  }).join('');

  const speciesTags = (dets || []).filter(d => d.species).map(d =>
    `<span style="display:inline-block;margin:3px 4px;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:600;background:#00b4a011;color:#00b4a0;border:1px solid #00b4a055;">
      ↳ Species: ${d.species.toUpperCase()} (${Math.round(d.species_confidence * 100)}%)
    </span>`
  ).join('');

  const imgTag = imgSrc
    ? `<img src="${imgSrc}" alt="Frame ${pageNum}" style="width:100%;height:auto;display:block;border-radius:6px;border:1px solid #e0e0e0;"/>`
    : `<div style="height:180px;border:1px solid #e0e0e0;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:13px;">No image available</div>`;

  return `
<div style="page-break-before:always;padding:28px 36px;font-family:'Inter',sans-serif;max-width:800px;margin:0 auto;">
  <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e0e0e0;padding-bottom:10px;margin-bottom:18px;">
    <span style="font-size:13px;font-weight:600;color:#0b1e2d;">Image ${pageNum}${label ? ' — ' + label : ''}</span>
    <span style="font-size:11px;color:#888;">${(dets || []).length} finding${(dets || []).length !== 1 ? 's' : ''}</span>
  </div>
  <div style="margin-bottom:16px;">${imgTag}</div>
  <div style="background:#f8fafb;border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;">
    <div style="font-size:12px;font-weight:600;color:#0b1e2d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Detection Results</div>
    <div>${tags || '<span style="color:#aaa;font-size:12px;">No detections</span>'}</div>
    ${speciesTags ? `<div style="margin-top:6px;">${speciesTags}</div>` : ''}
  </div>
</div>`;
}

/* ── Contact form submission ──────────────────── */
window.submitContactForm = async function () {
  const btn = document.getElementById('btn-submit-contact');
  const fields = ['f-first', 'f-last', 'f-email', 'f-company', 'f-use', 'f-message'];
  const payload = {};
  const map = { 'f-first': 'first_name', 'f-last': 'last_name', 'f-email': 'email',
                'f-company': 'company', 'f-use': 'use_case', 'f-message': 'message' };

  for (const id of fields) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (el.required && !el.value.trim()) {
      el.style.borderColor = '#e74c3c';
      el.focus();
      return;
    }
    el.style.borderColor = '';
    payload[map[id]] = el.value.trim();
  }

  if (btn) { btn.textContent = 'Sending…'; btn.disabled = true; }

  try {
    const API = window.API_BASE || 'http://localhost:8000';
    const r = await fetch(`${API}/contact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (btn) { btn.textContent = 'Sent ✓'; btn.style.background = '#00b4a0'; }
    fields.forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
  } catch (e) {
    if (btn) { btn.textContent = 'Send Request'; btn.disabled = false; }
    alert('Submission failed: ' + e.message);
  }
};
