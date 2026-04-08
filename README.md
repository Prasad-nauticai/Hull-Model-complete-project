# NautiCAI — Underwater Hull Inspection Platform

AI-powered underwater hull inspection using multi-model computer vision. Upload ROV/drone images or videos to detect corrosion, biofouling, debris, and anomalies — then generate professional inspection reports.

---

## Project Structure

```
nauticai/
├── api.py                          # FastAPI backend — multi-model pipeline (MAIN)
├── serve.py                        # Frontend dev server (port 3000)
├── index.html                      # Main UI — single page app
├── admin.html                      # Admin dashboard (inspection history)
├── .env.example                    # Environment variables template
├── requirements_api.txt            # Python dependencies
├── Dockerfile                      # Container build
│
├── js/
│   ├── dashboard.js                # Upload, detection, gallery, 3D view, job polling
│   ├── main.js                     # Navbar, animations, scroll effects
│   └── report.js                   # Dynamic PDF report generation
│
├── css/
│   └── style.css                   # Full design system (glassmorphism, dark theme)
│
├── backend_deploy_repo/
│   ├── hull_inspection_best.pt     # YOLOv8s — corrosion, marine_growth, debris (88.2% mAP)
│   ├── biofouling_best.pt          # YOLOv8 biofouling specialist (82.3% mAP)
│   ├── final_hull_best.onnx        # ONNX export (for TensorRT conversion)
│   ├── requirements.txt            # Minimal clean dependencies
│   └── Dockerfile
│
├── convert_model.py                # PT → ONNX → TensorRT conversion script
└── image.png                       # NautiCAI logo
```

> **Model files not included in repo** (too large for GitHub).
> Download links and setup instructions below.

---

## AI Pipeline

```
Image / Video Frame
        │
        ├─► hull_inspection_best.pt  (YOLOv8s, 88.2% mAP)
        │     → corrosion, marine_growth→biofouling, debris
        │
        ├─► best (3).pt              (YOLOv8m, 70.7% mAP)
        │     → hull→corrosion, biofouling, anomaly
        │
        └─► biofouling_best.pt       (YOLOv8, 82.3% mAP)
              → biofouling (specialist)

All detections merged → cross-model IoU deduplication
biofouling detections → ResNet-50 → algae / barnacles / mussels
```

---

## Detection Classes

| Class | Color | Source |
|---|---|---|
| `corrosion` | 🔴 Red | hull_inspection + general_hull |
| `biofouling` | 🟡 Amber | all 3 models |
| `debris` | 🟢 Green | hull_inspection |
| `anomaly` | 🟣 Purple | general_hull |

**Species classification** (on biofouling crops via ResNet-50):
`algae` · `barnacles` · `mussels`

---

## Model Files

| Model | Classes | mAP50 | Size | Download |
|---|---|---|---|---|
| `hull_inspection_best.pt` | corrosion, marine_growth, debris, healthy_surface | 88.2% | 22.5 MB | See releases |
| `best (3).pt` | hull, biofouling, anomaly | 70.7% | 52 MB | See releases |
| `biofouling_best.pt` | biofouling | 82.3% | 44 MB | See releases |
| `resnet50_species_full_model.pt` | algae, barnacles, mussels | — | 94 MB | See releases |

Place model files in the root directory before running.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Prasad-nauticai/Hull-Model-complete-project.git
cd Hull-Model-complete-project
pip install -r requirements_api.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

### 3. Download models

Download from GitHub Releases and place in root:
- `hull_inspection_best.pt`
- `best (3).pt`
- `biofouling_best.pt`
- `resnet50_species_full_model.pt`

### 4. Start backend (port 8000)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start frontend (port 3000)

```bash
python serve.py
```

### 6. Open browser

```
http://localhost:3000
```

---

## Docker

```bash
docker build -t nauticai .
docker run -p 8000:8000 --env-file .env nauticai
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/detect` | POST | Single image → immediate result. Video → returns job_id |
| `/job/{id}/status` | GET | Poll video job progress + results |
| `/batch-job` | POST | Submit multiple files async |
| `/inspections` | GET | Inspection history from Supabase |
| `/health` | GET | Model status, classes, thresholds |
| `/model-info` | GET | Full pipeline info |
| `/contact` | POST | Enterprise contact form |

---

## Video Processing

Smart frame sampling — always targets ~60 frames regardless of video length:

| Duration | Interval | Frames |
|---|---|---|
| 2 min | 2s | 60 |
| 5 min | 5s | 60 |
| 10 min | 10s | 60 |
| 20 min | 20s | 60 |
| 30 min | 30s | 60 |

Override: set `FRAME_INTERVAL_SEC=5` in `.env`

---

## Environment Variables

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
POSTGRES_DSN=postgresql://postgres:password@db.supabase.co:5432/postgres

# Optional overrides
YOLO_CONF=0.25
YOLO_IOU=0.30
YOLO_IMGSZ=416
FRAME_INTERVAL_SEC=   # leave empty for auto
N_WORKERS=4
BATCH_SIZE=8
```

---

## Tech Stack

- **Backend**: FastAPI, PyTorch, Ultralytics YOLOv8, OpenCV, asyncio
- **Frontend**: Vanilla JS, Canvas API, HTML/CSS (no frameworks)
- **Database**: Supabase (Postgres + object storage)
- **Models**: YOLOv8s, YOLOv8m, YOLOv8 specialist, ResNet-50
- **Deployment**: Docker, supports NVIDIA Jetson (TensorRT auto-detect)

---

## Supabase Schema

```sql
-- Inspections table
create table inspections (
  id uuid default gen_random_uuid() primary key,
  inspection_id text,
  file_name text,
  detected_classes text[],
  highest_confidence float,
  risk_level text,
  inference_time float,
  precision float,
  recall float,
  map50 float,
  map5095 float,
  image_url text,
  annotated_image_url text,
  status text,
  created_at timestamptz default now()
);

-- Enterprise contacts
create table enterprise_contacts (
  id uuid default gen_random_uuid() primary key,
  first_name text,
  last_name text,
  email text,
  company text,
  use_case text,
  message text,
  created_at timestamptz default now()
);
```

---

## License

MIT
