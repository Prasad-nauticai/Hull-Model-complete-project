"""
NautiCAI FastAPI Backend — High-Throughput Edition
====================================================
• Supports NVIDIA Jetson via TensorRT (.engine) auto-detection
• Async worker pool for parallel batch processing
• YOLO batch inference (N images per call) + FP16 on CUDA/Jetson
• POST /batch-job  → enqueue many files, returns job_id immediately
• GET  /job/{job_id}/status → poll live progress
• POST /detect     → single-file (unchanged, backward-compat)

Start locally:
    uvicorn api:app --reload --port 8000
On Jetson:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import io
import uuid
import time
import random
import base64
import asyncio
import tempfile
import shutil
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from collections import defaultdict

# ── Torch safe loading fix — MUST run before ultralytics import ──────────────
import torch

_orig_torch_load = torch.load
def _safe_load(*args, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*args, **kw)
torch.load = _safe_load

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ultralytics import YOLO
from supabase import create_client
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY  = os.getenv("SUPABASE_KEY", "").strip()
POSTGRES_DSN  = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")

# Parallel workers for batch job processing
N_WORKERS  = int(os.getenv("N_WORKERS",  "4"))
# Images per YOLO batch call (tune to VRAM/RAM)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# ── Smart frame interval — auto-scales with video duration ───────────────────
# Override with env FRAME_INTERVAL_SEC=N to force a fixed interval.
# If not set, _smart_frame_interval(duration_sec) is used at runtime.
FRAME_INTERVAL_OVERRIDE = os.getenv("FRAME_INTERVAL_SEC")   # None = auto

def _smart_frame_interval(duration_sec: float) -> float:
    """
    Target ~60 frames for any video length — gives consistent coverage.
    Override with env FRAME_INTERVAL_SEC=N to force a fixed interval.

    Examples:
      2 min  (120s)  → 120/60 = 2s interval  → 60 frames
      5 min  (300s)  → 300/60 = 5s interval  → 60 frames
      10 min (600s)  → 600/60 = 10s interval → 60 frames
      20 min (1200s) → 1200/60= 20s interval → 60 frames
      30 min (1800s) → 1800/60= 30s interval → 60 frames

    Hard limits:
      min interval = 2s  (avoid near-duplicate frames)
      max interval = 30s (avoid missing long defect-free stretches)
      max frames   = 120 (CPU safety cap)
    """
    if FRAME_INTERVAL_OVERRIDE:
        return float(FRAME_INTERVAL_OVERRIDE)

    TARGET_FRAMES = 60
    interval = duration_sec / TARGET_FRAMES

    # Clamp between 2s and 30s
    interval = max(2.0, min(30.0, interval))

    return round(interval, 1)

# ── Model resolution: prefer TensorRT .engine, then .pt ──────────────────────
_BASE   = os.path.dirname(os.path.abspath(__file__))
_DEPLOY = os.path.join(_BASE, "backend_deploy_repo")
_PROD   = os.path.join(_BASE, "nauticai_production")

def _find_model(stem: str) -> str:
    """Return first existing path: TensorRT engine > PyTorch .pt
    Searches root, backend_deploy_repo, nauticai_production."""
    candidates = [
        os.path.join(_BASE,   stem + ".engine"),
        os.path.join(_DEPLOY, stem + ".engine"),
        os.path.join(_PROD,   stem + ".engine"),
        os.path.join(_BASE,   stem + ".pt"),
        os.path.join(_DEPLOY, stem + ".pt"),
        os.path.join(_PROD,   stem + ".pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[MODEL] Found: {p}")
            return p
    raise FileNotFoundError(
        f"Model '{stem}.pt' not found in root, backend_deploy_repo, or nauticai_production"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-MODEL PIPELINE
#  Three complementary YOLO models run on every image/frame, detections are
#  merged and deduplicated by IoU overlap.
#
#  Model 1 — hull_inspection_best.pt  (YOLOv8s, 89% prec, 88.2% mAP50)
#    Classes: corrosion, marine_growth, debris, healthy_surface
#
#  Model 2 — best (2).pt  (YOLOv8s, 89% prec, 88.2% mAP50)  [same as M1 — skip if identical path]
#    Classes: corrosion, marine_growth, debris, healthy_surface
#
#  Model 3 — best (3).pt  (YOLOv8m, 80.7% prec, 70.7% mAP50)
#    Classes: hull→corrosion, biofouling, anomaly
#
#  Model 4 — biofouling_best.pt  (YOLOv8, 86.7% prec, 82.3% mAP50)
#    Classes: biofouling  (specialist — highest confidence for fouling)
#
#  ResNet-50 — species classifier on every biofouling detection
#    Classes: algae, barnacles, mussels
# ══════════════════════════════════════════════════════════════════════════════

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = torch.cuda.is_available()
print(f"[DEVICE] {device}  |  FP16={USE_HALF}")

# ── Load ResNet species classifier ───────────────────────────────────────────
def _find_resnet() -> str:
    for p in [
        os.path.join(_BASE,   "resnet50_species_full_model.pt"),
        os.path.join(_DEPLOY, "resnet50_species_full_model.pt"),
        os.path.join(_PROD,   "resnet50_species_full_model.pt"),
    ]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("resnet50_species_full_model.pt not found")

RESNET_MODEL_PATH = _find_resnet()
SPECIES_CLASSES   = ["algae", "barnacles", "mussels"]

try:
    import torchvision.transforms as transforms
    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    resnet_model = torch.load(RESNET_MODEL_PATH, map_location=device)
    if USE_HALF:
        resnet_model = resnet_model.half()
    resnet_model.eval()
    print(f"[MODEL] ResNet loaded ✓  ({RESNET_MODEL_PATH})")
except Exception as e:
    print(f"[WARN] ResNet load failed: {e}")
    resnet_model = None
    resnet_transform = None

# ── Load all YOLO models ──────────────────────────────────────────────────────
def _load_yolo(stem: str, label: str) -> Optional[YOLO]:
    try:
        path = _find_model(stem)
        m = YOLO(path)
        if USE_HALF and not path.endswith(".engine"):
            m.model = m.model.half()
        print(f"[MODEL] {label} loaded ✓  classes={list(m.names.values())}")
        return m
    except Exception as e:
        print(f"[WARN] {label} ({stem}) not loaded: {e}")
        return None

# Primary — hull inspection specialist (89% mAP50)
model_hull   = _load_yolo("hull_inspection_best", "HullInspection")

# Secondary — general hull + biofouling (70.7% mAP50, catches biofouling + anomaly)
model_gen    = _load_yolo("best (3)", "GeneralHull")

# Biofouling specialist — highest confidence for marine growth (82.3% mAP50)
model_bio    = _load_yolo("biofouling_best", "BiofoulingSpec")

# Fallback primary path (used for /health and metrics reporting)
_primary_stem = "hull_inspection_best"
try:
    MODEL_PATH = _find_model(_primary_stem)
except Exception:
    MODEL_PATH = "none"

# ── Aggregate model metrics (weighted average of loaded models) ───────────────
def _load_model_metrics(model_path: str) -> dict:
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            m = ckpt.get("train_metrics") or {}
            return {
                "precision": round(m.get("metrics/precision(B)", 0), 4) or None,
                "recall":    round(m.get("metrics/recall(B)",    0), 4) or None,
                "map50":     round(m.get("metrics/mAP50(B)",     0), 4) or None,
                "map5095":   round(m.get("metrics/mAP50-95(B)",  0), 4) or None,
            }
    except Exception:
        pass
    return {"precision": None, "recall": None, "map50": None, "map5095": None}

# Report metrics from the best model (hull_inspection_best)
_primary_path = _find_model("hull_inspection_best") if model_hull else MODEL_PATH
MODEL_METRICS = _load_model_metrics(_primary_path)
print(f"[MODEL] Pipeline metrics (primary): {MODEL_METRICS}")

# ── Class remaps — normalize across all 3 models to unified names ─────────────
CLASS_NAME_REMAP = {
    "hull":            "corrosion",     # best(3) → unified
    "marine_growth":   "biofouling",    # hull_inspection → unified
    "anomaly":         "anomaly",
}

# Classes to completely suppress from output — not useful for inspection reports
CLASS_SUPPRESS = {"healthy_surface"}

def remap_class(name: str) -> str:
    return CLASS_NAME_REMAP.get(name.lower().strip(), name.lower().strip())

# ── Unified class colors (BGR for OpenCV) ─────────────────────────────────────
CLASS_COLORS_BGR = {
    "corrosion":   (60,  76, 231),   # red
    "biofouling":  (0,  165, 240),   # amber
    "debris":      (80, 180,  80),   # green
    "anode":       (34, 200, 230),   # cyan
    "paint_peel":  (60,  76, 231),   # red
    "anomaly":     (180, 60, 220),   # purple
}
DEFAULT_COLOR_BGR = (0, 200, 176)

# ── Inference thresholds ─────────────────────────────────────────────────────
# YOLO_CONF  — min confidence per model. 0.25 = good balance on CPU.
# YOLO_IOU   — NMS overlap. 0.30 removes duplicates without merging real defects.
# MERGE_IOU  — cross-model dedup threshold. Boxes with IoU > this are merged.
# YOLO_IMGSZ — 416 is ~40% faster than 640 on CPU, minimal accuracy loss.
YOLO_CONF  = float(os.getenv("YOLO_CONF",  "0.25"))
YOLO_IOU   = float(os.getenv("YOLO_IOU",   "0.30"))
MERGE_IOU  = float(os.getenv("MERGE_IOU",  "0.40"))  # cross-model dedup
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ",   "416"))


# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="NautiCAI Detection API — HT Edition", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thread pool for blocking CPU work (frame extraction, drawing) ─────────────
_thread_pool = ThreadPoolExecutor(max_workers=N_WORKERS * 2)

# ── Supabase ──────────────────────────────────────────────────────────────────
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected ✓")
except Exception as e:
    print(f"[WARN] Supabase: {e}")
    supabase = None


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC JOB QUEUE + STORE
# ══════════════════════════════════════════════════════════════════════════════
job_queue: asyncio.Queue = None   # created in startup
# job_store schema per job_id:
#   status:          "queued" | "running" | "done" | "error"
#   total_files:     int
#   processed_files: int
#   current_file:    str
#   detections_so_far: int
#   started_at:      float (epoch)
#   elapsed_sec:     float
#   results:         dict | None   (populated when done)
#   error:           str | None
job_store: dict = {}


async def _worker_loop():
    """Persistent coroutine that pulls jobs from the queue and processes them."""
    while True:
        job_id, tmp_dir, file_paths = await job_queue.get()
        try:
            await _process_batch_job(job_id, tmp_dir, file_paths)
        except Exception as e:
            job_store[job_id]["status"] = "error"
            job_store[job_id]["error"]  = str(e)
            print(f"[JOB] {job_id} FAILED: {e}")
        finally:
            # Cleanup temp dir
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            job_queue.task_done()


@app.on_event("startup")
async def startup_event():
    global job_queue
    job_queue = asyncio.Queue()
    for i in range(N_WORKERS):
        asyncio.create_task(_worker_loop())
    print(f"[WORKERS] {N_WORKERS} async workers started")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_color(class_name: str):
    return CLASS_COLORS_BGR.get(class_name.lower().strip(), DEFAULT_COLOR_BGR)


def draw_boxes(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    img = image_bgr.copy()
    H, W = img.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = det["class_name"]
        conf  = det["confidence"]
        color = get_color(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Corner markers
        cs = 12
        for (cx, cy, dx, dy) in [
            (x1, y1,  cs,  cs), (x2, y1, -cs,  cs),
            (x1, y2,  cs, -cs), (x2, y2, -cs, -cs),
        ]:
            cv2.line(img, (cx, cy), (cx + dx, cy), color, 3)
            cv2.line(img, (cx, cy), (cx, cy + dy), color, 3)
        # Label
        tag = f"{label}  {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.48, 1
        (tw, th), _ = cv2.getTextSize(tag, font, scale, thick)
        lx, ly = x1, max(y1 - 6, th + 4)
        cv2.rectangle(img, (lx, ly - th - 4), (lx + tw + 10, ly + 2), color, -1)
        cv2.putText(img, tag, (lx + 5, ly - 2), font, scale, (10, 10, 10), thick, cv2.LINE_AA)
    # Detection count overlay
    clabel = f"{len(detections)} detection{'s' if len(detections) != 1 else ''} found"
    (cw, ch), _ = cv2.getTextSize(clabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (W - cw - 20, H - ch - 16), (W - 4, H - 4), (6, 19, 32), -1)
    cv2.putText(img, clabel, (W - cw - 14, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 176), 1, cv2.LINE_AA)
    return img


def img_to_b64(image_bgr: np.ndarray) -> str:
    success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _top_per_class(detections: list, max_per_class: int = 2) -> list:
    buckets = defaultdict(list)
    for d in detections:
        buckets[d["class_name"]].append(d)
    result = []
    for cls_dets in buckets.values():
        cls_dets.sort(key=lambda x: x["confidence"], reverse=True)
        result.extend(cls_dets[:max_per_class])
    return result


def _run_resnet_species(image_bgr: np.ndarray, x1, y1, x2, y2) -> tuple:
    """Crop + classify species. Returns (species_name, confidence)."""
    if resnet_model is None:
        return None, None
    try:
        h, w = image_bgr.shape[:2]
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        crop_rgb = cv2.cvtColor(image_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        inp = resnet_transform(crop_rgb).unsqueeze(0).to(device)
        if USE_HALF:
            inp = inp.half()
        with torch.no_grad():
            probs = torch.nn.functional.softmax(resnet_model(inp)[0], dim=0)
            top_prob, top_idx = torch.max(probs, 0)
        return SPECIES_CLASSES[top_idx.item()], float(round(top_prob.item(), 4))
    except Exception as e:
        print(f"[ResNet] species error: {e}")
        return None, None


def _parse_detections(per_model_results, image_bgr: np.ndarray) -> list:
    """
    Accepts list of YOLO Result objects (one per model) for a single image.
    Merges, deduplicates, runs ResNet on biofouling crops.
    """
    return _merge_detections(per_model_results, image_bgr)


def _calc_risk(detections: list):
    """
    Risk is based on peak detection confidence from the model:
      HIGH   — max confidence > 0.40
      MEDIUM — max confidence > 0.25
      LOW    — max confidence > 0.15
      SAFE   — no detections
    """
    if not detections:
        return "SAFE", 0.0, 0.0
    max_conf = max(d["confidence"] for d in detections)
    avg_conf = round(sum(d["confidence"] for d in detections) / len(detections), 4)
    risk = "HIGH" if max_conf > 0.40 else ("MEDIUM" if max_conf > 0.25 else "LOW")
    return risk, max_conf, avg_conf


def upload_to_supabase(bucket: str, filename: str, data: bytes, content_type: str):
    if supabase is None:
        return None
    try:
        unique_name = f"{uuid.uuid4()}_{filename}"
        supabase.storage.from_(bucket).upload(
            unique_name, data, file_options={"content-type": content_type}
        )
        return supabase.storage.from_(bucket).get_public_url(unique_name)
    except Exception as e:
        print(f"[Supabase] upload error: {e}")
        return None


def _save_to_supabase(inspection_id, filename, detections, max_conf, risk,
                      inference_ms, annotated_bgr, raw_bytes, content_type):
    try:
        _, jpg_buf = cv2.imencode(".jpg", annotated_bgr)
        annotated_url = upload_to_supabase(
            "image_bucket", f"annotated_{inspection_id}.jpg",
            jpg_buf.tobytes(), "image/jpeg")
        original_url = upload_to_supabase(
            "image_bucket", f"original_{inspection_id}.jpg",
            raw_bytes, content_type)
        if supabase is not None:
            supabase.table("inspections").insert({
                "inspection_id":       inspection_id,
                "file_name":           filename,
                "detected_classes":    list(set(d["class_name"] for d in detections)),
                "highest_confidence":  float(max_conf),
                "risk_level":          risk,
                "inference_time":      inference_ms / 1000,
                "precision":           MODEL_METRICS["precision"],
                "recall":              MODEL_METRICS["recall"],
                "map50":               MODEL_METRICS["map50"],
                "map5095":             MODEL_METRICS["map5095"],
                "image_url":           original_url,
                "annotated_image_url": annotated_url,
                "status":              "completed",
            }).execute()
    except Exception as e:
        print(f"[Supabase] DB insert error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH YOLO INFERENCE — N images in one GPU call
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-MODEL BATCH INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def _run_one_model(yolo_model: YOLO, image_paths: list, conf_override: float = None) -> list:
    """Run a single YOLO model synchronously. Returns list of results."""
    return yolo_model.predict(
        source=image_paths,
        conf=conf_override if conf_override is not None else YOLO_CONF,
        iou=YOLO_IOU,
        agnostic_nms=True,
        imgsz=YOLO_IMGSZ,
        save=False,
        verbose=False,
        half=USE_HALF,
        stream=False,
    )


async def _yolo_batch_predict(image_paths: list) -> list:
    """
    Run all loaded YOLO models on the same batch of images.
    Each model uses its own confidence threshold:
      hull_inspection + general_hull → YOLO_CONF (0.25)
      biofouling_spec                → 0.15 (single-class, scores lower naturally)
    Returns list of merged detection lists — one per image.
    """
    loop   = asyncio.get_event_loop()
    # (model, label, conf_override)
    active = [
        (model_hull, "hull",  YOLO_CONF),
        (model_gen,  "gen",   YOLO_CONF),
        (model_bio,  "bio",   0.15),       # biofouling specialist — lower threshold
    ]
    active = [(m, lbl, c) for m, lbl, c in active if m is not None]

    # Run each model concurrently in thread pool
    tasks = [
        loop.run_in_executor(_thread_pool, _run_one_model, m, image_paths, c)
        for m, _, c in active
    ]
    all_model_results = await asyncio.gather(*tasks)

    # Merge per-image
    merged = []
    for img_idx in range(len(image_paths)):
        per_model = [all_model_results[mi][img_idx] for mi in range(len(active))]
        merged.append(per_model)
    return merged


def _iou(a: dict, b: dict) -> float:
    """Compute IoU between two detection dicts."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def _merge_detections(per_model_results: list, image_bgr) -> list:
    """
    Parse detections from all models, merge and deduplicate by IoU.
    Higher-confidence detection wins when two boxes overlap > MERGE_IOU.
    """
    all_dets = []

    for result in per_model_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs   = result.boxes.conf.cpu().numpy()
        xyxy    = result.boxes.xyxy.cpu().numpy()

        for i in range(len(cls_ids)):
            raw_name   = result.names[cls_ids[i]]
            class_name = remap_class(raw_name)

            # Skip suppressed classes (e.g. healthy_surface — not a defect)
            if class_name in CLASS_SUPPRESS or raw_name.lower() in CLASS_SUPPRESS:
                continue

            x1, y1, x2, y2 = xyxy[i]
            det = {
                "class_name": class_name,
                "confidence": float(round(confs[i], 4)),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "_raw": raw_name,
            }
            all_dets.append(det)

    # Sort by confidence descending — greedy NMS across models
    all_dets.sort(key=lambda d: d["confidence"], reverse=True)

    # Cross-model dedup: suppress lower-conf box if IoU > MERGE_IOU
    kept = []
    for det in all_dets:
        overlap = any(_iou(det, k) > MERGE_IOU for k in kept)
        if not overlap:
            kept.append(det)

    # Run ResNet species on biofouling detections
    for det in kept:
        raw = det.pop("_raw", det["class_name"])
        is_bio = "biofouling" in raw.lower() or "fouling" in raw.lower() \
                 or "marine_growth" in raw.lower()
        if is_bio:
            sp, sp_conf = _run_resnet_species(image_bgr,
                                               det["x1"], det["y1"],
                                               det["x2"], det["y2"])
            if sp:
                det["species"]            = sp
                det["species_confidence"] = sp_conf

    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO FRAME EXTRACTION — sequential read (fast), batch YOLO inference
# ══════════════════════════════════════════════════════════════════════════════
def _extract_frames_sequential(video_path: str, frame_indices: list, fps: float) -> list:
    """
    Extract frames in one sequential pass — much faster than re-opening cap per frame.
    Returns list of (frame_bgr, frame_idx, timestamp_sec).
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    idx_set = set(frame_indices)
    current = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in idx_set:
            results.append((frame, current, current / fps))
        current += 1
        if current > max(frame_indices):
            break
    cap.release()
    return results


async def _detect_video_frames(filename: str, raw_bytes: bytes,
                                job_id: str = None) -> dict:
    """
    Optimized video pipeline:
    1. Extract all frames in one sequential pass
    2. Resize all frames in parallel (thread pool)
    3. Run ALL frames through all 3 YOLO models in ONE big batch each — no chunking
       (3 models run concurrently via asyncio.gather)
    4. Parse + annotate all frames in parallel (thread pool)
    5. ResNet species on biofouling crops only
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
        vf.write(raw_bytes)
        vf_path = vf.name

    try:
        cap = cv2.VideoCapture(vf_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration_sec    = total_frames / fps
        interval        = _smart_frame_interval(duration_sec)
        step            = max(1, int(fps * interval))
        frame_indices   = list(range(0, total_frames, step))
        total_to_sample = len(frame_indices)

        print(f"[VIDEO] {filename}  dur={duration_sec:.0f}s  "
              f"fps={fps:.1f}  interval={interval}s  frames={total_to_sample}")

        loop = asyncio.get_event_loop()

        # ── Step 1: Extract all frames in one sequential pass ─────────────
        if job_id and job_id in job_store:
            job_store[job_id]["current_file"] = f"{filename} — extracting {total_to_sample} frames..."
        frame_data = await loop.run_in_executor(
            _thread_pool,
            lambda: _extract_frames_sequential(vf_path, frame_indices, fps)
        )
        frame_data = [(f, idx, ts) for f, idx, ts in frame_data if f is not None]
        if not frame_data:
            raise HTTPException(400, "No frames could be extracted from video")

        # ── Step 2: Resize all frames + write to temp files in parallel ────
        def _resize_write(item):
            frame_bgr, fp = item
            h, w = frame_bgr.shape[:2]
            if max(h, w) > YOLO_IMGSZ:
                scale     = YOLO_IMGSZ / max(h, w)
                frame_bgr = cv2.resize(frame_bgr,
                                       (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(fp, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return frame_bgr   # return resized frame for annotation later

        tmp_paths = [
            os.path.join(tempfile.gettempdir(), f"ncat_{uuid.uuid4().hex}.jpg")
            for _ in frame_data
        ]
        resized_frames = await loop.run_in_executor(
            _thread_pool,
            lambda: list(_thread_pool.map(_resize_write,
                                          [(fd[0], tp) for fd, tp in zip(frame_data, tmp_paths)]))
        )

        # ── Step 3: Run ALL frames through all 3 models in ONE batch each ──
        # All 3 models run concurrently — total time = slowest model, not sum
        if job_id and job_id in job_store:
            job_store[job_id]["current_file"] = f"{filename} — running inference on {total_to_sample} frames..."

        active_models = [
            (model_hull, YOLO_CONF),
            (model_gen,  YOLO_CONF),
            (model_bio,  0.15),
        ]
        active_models = [(m, c) for m, c in active_models if m is not None]

        t0 = time.time()
        # Fire all models at once on the full frame list
        model_tasks = [
            loop.run_in_executor(_thread_pool, _run_one_model, m, tmp_paths, c)
            for m, c in active_models
        ]
        all_model_outputs = await asyncio.gather(*model_tasks)
        total_inf_ms = round((time.time() - t0) * 1000, 1)

        # ── Step 4: Cleanup temp files ─────────────────────────────────────
        for fp in tmp_paths:
            try: os.unlink(fp)
            except Exception: pass

        # ── Step 5: Per-frame — merge detections + annotate in parallel ────
        if job_id and job_id in job_store:
            job_store[job_id]["current_file"] = f"{filename} — annotating frames..."

        def _process_frame(args):
            i, (orig_bgr, frame_idx, timestamp_sec) = args
            resized_bgr = resized_frames[i]
            # Collect this frame's results from each model
            per_model = [all_model_outputs[mi][i] for mi in range(len(active_models))]
            dets = _merge_detections(per_model, resized_bgr)
            dets.sort(key=lambda d: d["confidence"], reverse=True)
            dets = dets[:10]

            mins     = int(timestamp_sec // 60)
            secs     = int(timestamp_sec % 60)
            ts_label = f"{mins:02d}:{secs:02d}"
            for det in dets:
                det["timestamp_label"] = ts_label
                det["timestamp_sec"]   = round(timestamp_sec, 2)

            annotated = draw_boxes(resized_bgr, dets)
            return {
                "frame_index":     frame_idx,
                "timestamp_sec":   round(timestamp_sec, 2),
                "timestamp_label": ts_label,
                "filename":        f"frame_{mins:02d}m{secs:02d}s.jpg",
                "image":           img_to_b64(annotated),
                "detections":      dets,
                "summary": {
                    "total":             len(dets),
                    "inference_time_ms": round(total_inf_ms / max(len(frame_data), 1), 1),
                },
            }

        frame_results = await loop.run_in_executor(
            _thread_pool,
            lambda: list(_thread_pool.map(_process_frame, enumerate(frame_data)))
        )

        # Update job progress
        if job_id and job_id in job_store:
            job_store[job_id]["progress_pct"] = 95
            job_store[job_id]["elapsed_sec"]  = round(time.time() - job_store[job_id]["started_at"], 1)

        all_detections = [det for fr in frame_results for det in fr["detections"]]

    finally:
        try: os.unlink(vf_path)
        except Exception: pass

    risk, max_conf, avg_conf = _calc_risk(all_detections)
    inspection_id = f"NCR-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

    if frame_results:
        first_bytes = base64.b64decode(frame_results[0]["image"].split(",")[1])
        first_bgr   = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
        _save_to_supabase(inspection_id, filename, all_detections, max_conf, risk,
                          total_inf_ms, first_bgr, raw_bytes[:1024], "video/mp4")

    print(f"[VIDEO] Done  frames={len(frame_results)}  dets={len(all_detections)}  "
          f"inf={total_inf_ms}ms")

    return {
        "inspection_id":    inspection_id,
        "is_video":         True,
        "video_frames":     frame_results,
        "annotated_images": frame_results,
        "detections":       all_detections,
        "annotated_image":  frame_results[0]["image"] if frame_results else None,
        "summary": {
            "total":              len(all_detections),
            "risk_level":         risk,
            "avg_confidence":     avg_conf,
            "max_confidence":     float(max_conf),
            "inference_time_ms":  total_inf_ms,
            "frames_analyzed":    len(frame_results),
            "video_duration_sec": round(duration_sec, 1),
            "frame_interval_sec": interval,
        },
        "model_metrics": MODEL_METRICS,
        "timestamp":      datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH JOB PROCESSOR (runs inside worker coroutine)
# ══════════════════════════════════════════════════════════════════════════════
async def _process_batch_job(job_id: str, tmp_dir: str, file_paths: list[dict]):
    """
    Process all files for a batch job.
    file_paths: list of {"path": str, "filename": str, "is_video": bool}
    Updates job_store[job_id] in real time.
    """
    store = job_store[job_id]
    store["status"]    = "running"
    store["started_at"] = time.time()

    all_detections     = []
    all_annotated      = []   # [{filename, image, detections, summary}]
    total_inf_ms       = 0.0

    image_files = [f for f in file_paths if not f["is_video"]]
    video_files = [f for f in file_paths if f["is_video"]]

    # ── Process images in BATCH_SIZE chunks ──────────────────────────────────
    for chunk_start in range(0, len(image_files), BATCH_SIZE):
        chunk = image_files[chunk_start:chunk_start + BATCH_SIZE]
        chunk_paths = [f["path"] for f in chunk]

        store["current_file"] = chunk[0]["filename"]

        t0 = time.time()
        try:
            results = await _yolo_batch_predict(chunk_paths)
        except Exception as e:
            print(f"[BATCH] chunk inference error: {e}")
            store["processed_files"] += len(chunk)
            continue
        inf_ms = round((time.time() - t0) * 1000, 1)
        total_inf_ms += inf_ms

        # Parse each result
        for i, (result, finfo) in enumerate(zip(results, chunk)):
            loop = asyncio.get_event_loop()
            raw_bytes = await loop.run_in_executor(
                _thread_pool, lambda p=finfo["path"]: open(p, "rb").read()
            )
            nparr = np.frombuffer(raw_bytes, np.uint8)
            image_bgr = await loop.run_in_executor(
                _thread_pool,
                lambda n=nparr: cv2.imdecode(n, cv2.IMREAD_COLOR)
            )

            dets = _parse_detections(result, image_bgr)
            dets = _top_per_class(dets, max_per_class=2)
            dets.sort(key=lambda d: d["confidence"], reverse=True)

            annotated_bgr = await loop.run_in_executor(
                _thread_pool, lambda img=image_bgr, d=dets: draw_boxes(img, d)
            )
            annotated_b64 = await loop.run_in_executor(
                _thread_pool, lambda img=annotated_bgr: img_to_b64(img)
            )

            all_detections.extend(dets)
            all_annotated.append({
                "filename":   finfo["filename"],
                "image":      annotated_b64,
                "detections": dets,
                "summary": {"total": len(dets), "inference_time_ms": round(inf_ms / len(chunk), 1)},
            })

            store["processed_files"]    += 1
            store["detections_so_far"]  += len(dets)
            store["elapsed_sec"]         = round(time.time() - store["started_at"], 1)
            store["progress_pct"]        = int(
                store["processed_files"] / store["total_files"] * 100
            )

    # ── Process video files sequentially (each internally parallelized) ───────
    for finfo in video_files:
        store["current_file"] = finfo["filename"]
        try:
            raw_bytes = open(finfo["path"], "rb").read()
            vid_result = await _detect_video_frames(finfo["filename"], raw_bytes)
            all_detections.extend(vid_result.get("detections", []))
            for fr in vid_result.get("video_frames", []):
                fr["filename"] = fr.get("filename") or finfo["filename"]
                all_annotated.append({
                    "filename":   fr["filename"],
                    "image":      fr.get("image"),
                    "detections": fr.get("detections", []),
                    "summary":    fr.get("summary", {}),
                    "timestamp_label": fr.get("timestamp_label"),
                })
            total_inf_ms += vid_result.get("summary", {}).get("inference_time_ms", 0)
        except Exception as e:
            print(f"[BATCH-VIDEO] {finfo['filename']} error: {e}")
        store["processed_files"]   += 1
        store["detections_so_far"] += len(all_detections)
        store["elapsed_sec"]        = round(time.time() - store["started_at"], 1)
        store["progress_pct"]       = int(
            store["processed_files"] / store["total_files"] * 100
        )

    # ── Build final result ────────────────────────────────────────────────────
    risk, max_conf, avg_conf = _calc_risk(all_detections)
    inspection_id = f"NCR-BATCH-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    store["status"]       = "done"
    store["progress_pct"] = 100
    store["elapsed_sec"]  = round(time.time() - store["started_at"], 1)
    store["results"] = {
        "inspection_id":    inspection_id,
        "is_batch":         True,
        "detections":       all_detections,
        "annotated_images": all_annotated,
        "annotated_image":  all_annotated[0]["image"] if all_annotated else None,
        "summary": {
            "total":              len(all_detections),
            "risk_level":         risk,
            "avg_confidence":     avg_conf,
            "max_confidence":     float(max_conf),
            "inference_time_ms":  round(total_inf_ms, 1),
            "files_processed":    store["processed_files"],
        },
        "model_metrics": MODEL_METRICS,
        "timestamp":      datetime.now().isoformat(),
    }

    print(f"[JOB] {job_id} DONE  files={store['processed_files']}  "
          f"detections={len(all_detections)}  time={store['elapsed_sec']}s")


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /batch-job
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/batch-job")
async def submit_batch_job(files: List[UploadFile] = File(...)):
    """
    Submit multiple files for async batch inference.
    Returns immediately with a job_id.
    Poll GET /job/{job_id}/status for progress.
    """
    if not files:
        raise HTTPException(400, "No files provided")

    allowed = {
        "image/jpeg", "image/png", "image/jpg", "image/webp",
        "video/mp4", "video/quicktime", "video/avi",
    }

    job_id  = f"JOB-{uuid.uuid4().hex[:10].upper()}"
    tmp_dir = tempfile.mkdtemp(prefix=f"ncat_{job_id}_")
    file_paths = []

    for uf in files:
        if uf.content_type and uf.content_type not in allowed:
            continue  # skip unsupported silently
        suffix  = Path(uf.filename or "upload.jpg").suffix or ".jpg"
        fp      = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")
        content = await uf.read()
        with open(fp, "wb") as fh:
            fh.write(content)
        file_paths.append({
            "path":     fp,
            "filename": uf.filename or os.path.basename(fp),
            "is_video": (uf.content_type or "").startswith("video/"),
        })

    if not file_paths:
        raise HTTPException(400, "No supported files found in submission")

    # Initialise job store entry
    job_store[job_id] = {
        "status":           "queued",
        "total_files":      len(file_paths),
        "processed_files":  0,
        "current_file":     "",
        "detections_so_far": 0,
        "progress_pct":     0,
        "started_at":       None,
        "elapsed_sec":      0,
        "results":          None,
        "error":            None,
    }

    await job_queue.put((job_id, tmp_dir, file_paths))
    print(f"[JOB] {job_id} queued  files={len(file_paths)}")

    return JSONResponse({
        "job_id":       job_id,
        "total_files":  len(file_paths),
        "status":       "queued",
        "poll_url":     f"/job/{job_id}/status",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /job/{job_id}/status
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Poll this endpoint every 2s to track batch job progress."""
    if job_id not in job_store:
        raise HTTPException(404, f"Job '{job_id}' not found")
    entry = job_store[job_id]
    return JSONResponse({
        "job_id":            job_id,
        "status":            entry["status"],
        "total_files":       entry["total_files"],
        "processed_files":   entry["processed_files"],
        "progress_pct":      entry["progress_pct"],
        "current_file":      entry["current_file"],
        "detections_so_far": entry["detections_so_far"],
        "elapsed_sec":       entry["elapsed_sec"],
        "error":             entry["error"],
        # Only filled when status == "done"
        "results":           entry["results"],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /jobs  (list recent jobs)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/jobs")
async def list_jobs():
    """Returns all known jobs (most recent first by key order)."""
    jobs = [
        {
            "job_id":          jid,
            "status":          v["status"],
            "total_files":     v["total_files"],
            "processed_files": v["processed_files"],
            "progress_pct":    v["progress_pct"],
            "elapsed_sec":     v["elapsed_sec"],
        }
        for jid, v in reversed(list(job_store.items()))
    ]
    return JSONResponse({"jobs": jobs})


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /detect  (single file — backward compatible)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Single-file inference endpoint.
    - Images: returns result immediately.
    - Videos: submits async job, returns job_id immediately.
      Poll GET /job/{job_id}/status for live progress + final results.
    """
    allowed = {
        "image/jpeg", "image/png", "image/jpg", "image/webp",
        "video/mp4", "video/quicktime", "video/avi",
        "video/x-msvideo", "video/webm",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    raw_bytes = await file.read()
    is_video  = file.content_type and file.content_type.startswith("video/")

    # ── Video: submit as async job so frontend can poll progress ─────────────
    if is_video:
        job_id  = f"VID-{uuid.uuid4().hex[:10].upper()}"
        tmp_dir = tempfile.mkdtemp(prefix=f"ncat_{job_id}_")
        suffix  = Path(file.filename or "upload.mp4").suffix or ".mp4"
        fp      = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")
        with open(fp, "wb") as fh:
            fh.write(raw_bytes)

        job_store[job_id] = {
            "status":            "queued",
            "total_files":       1,
            "processed_files":   0,
            "current_file":      file.filename or "video",
            "detections_so_far": 0,
            "progress_pct":      0,
            "started_at":        None,
            "elapsed_sec":       0,
            "results":           None,
            "error":             None,
        }

        file_paths = [{
            "path":     fp,
            "filename": file.filename or "upload.mp4",
            "is_video": True,
        }]
        await job_queue.put((job_id, tmp_dir, file_paths))
        print(f"[VIDEO-JOB] {job_id} queued: {file.filename}")

        return JSONResponse({
            "job_id":      job_id,
            "status":      "queued",
            "is_video":    True,
            "poll_url":    f"/job/{job_id}/status",
            "message":     "Video queued for processing. Poll /job/{job_id}/status for progress.",
        })

    # ── Image: run synchronously and return immediately ───────────────────────
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        results = await _yolo_batch_predict([tmp_path])
    finally:
        os.unlink(tmp_path)

    inference_ms = round((time.time() - t0) * 1000, 1)
    result = results[0]

    nparr     = np.frombuffer(raw_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = _parse_detections(result, image_bgr)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = detections[:50]

    annotated_bgr = draw_boxes(image_bgr, detections)
    annotated_b64 = img_to_b64(annotated_bgr)

    risk, max_conf, avg_conf = _calc_risk(detections)
    inspection_id = f"NCR-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

    _save_to_supabase(inspection_id, file.filename, detections, max_conf, risk,
                      inference_ms, annotated_bgr, raw_bytes,
                      file.content_type or "image/jpeg")

    return JSONResponse({
        "inspection_id":   inspection_id,
        "is_video":        False,
        "detections":      detections,
        "annotated_image": annotated_b64,
        "summary": {
            "total":             len(detections),
            "risk_level":        risk,
            "avg_confidence":    avg_conf,
            "max_confidence":    float(max_conf),
            "inference_time_ms": inference_ms,
        },
        "model_metrics": MODEL_METRICS,
        "timestamp":      datetime.now().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /inspections
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/inspections")
async def get_inspections(limit: int = 20):
    if supabase is None:
        return JSONResponse({"inspections": [], "error": "Supabase not configured"})
    try:
        resp = (supabase.table("inspections")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute())
        return JSONResponse({"inspections": resp.data or []})
    except Exception as e:
        return JSONResponse({"inspections": [], "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /health
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    q_size = job_queue.qsize() if job_queue else -1

    # Collect all classes from all loaded models, remapped
    all_classes = []
    models_loaded = {}
    for lbl, m in [("hull_inspection", model_hull), ("general_hull", model_gen), ("biofouling_spec", model_bio)]:
        if m is not None:
            cls = [remap_class(c) for c in m.names.values()]
            all_classes.extend(c for c in cls if c not in all_classes)
            models_loaded[lbl] = list(m.names.values())

    return {
        "status":         "ok",
        "pipeline":       "multi-model",
        "models_loaded":  models_loaded,
        "device":         str(device),
        "fp16":           USE_HALF,
        "n_workers":      N_WORKERS,
        "batch_size":     BATCH_SIZE,
        "conf_threshold": YOLO_CONF,
        "iou_threshold":  YOLO_IOU,
        "merge_iou":      MERGE_IOU,
        "imgsz":          YOLO_IMGSZ,
        "queue_depth":    q_size,
        "active_jobs":    sum(1 for j in job_store.values() if j["status"] == "running"),
        "supabase":       supabase is not None,
        "classes":        all_classes,
        "metrics":        MODEL_METRICS,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /model-info  (detailed model info for frontend)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/model-info")
async def model_info():
    """Returns detailed model metadata for the frontend."""
    models_info = {}
    all_classes = []
    for lbl, m in [("hull_inspection", model_hull), ("general_hull", model_gen), ("biofouling_spec", model_bio)]:
        if m is not None:
            raw = list(m.names.values())
            remapped = [remap_class(c) for c in raw]
            models_info[lbl] = {"raw_classes": raw, "classes": remapped}
            all_classes.extend(c for c in remapped if c not in all_classes)
    return {
        "pipeline":       "multi-model",
        "models":         models_info,
        "device":         str(device),
        "fp16":           USE_HALF,
        "conf_threshold": YOLO_CONF,
        "iou_threshold":  YOLO_IOU,
        "merge_iou":      MERGE_IOU,
        "imgsz":          YOLO_IMGSZ,
        "all_classes":    all_classes,
        "class_remap":    CLASS_NAME_REMAP,
        "metrics":        MODEL_METRICS,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /contact
# ══════════════════════════════════════════════════════════════════════════════
class ContactPayload(BaseModel):
    first_name: str
    last_name: str
    email: str
    company: str = ""
    use_case: str = ""
    message: str = ""


def _insert_contact_postgres(payload: ContactPayload) -> bool:
    if not POSTGRES_DSN or psycopg2 is None:
        return False
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO enterprise_contacts
                   (first_name, last_name, email, company, use_case, message)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (payload.first_name, payload.last_name, payload.email,
                 payload.company, payload.use_case, payload.message),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Postgres] contact insert error: {e}")
        return False
    finally:
        if conn:
            conn.close()


@app.post("/contact")
async def submit_contact(payload: ContactPayload):
    supabase_ok = False
    if supabase is not None:
        try:
            supabase.table("enterprise_contacts").insert({
                "first_name": payload.first_name,
                "last_name":  payload.last_name,
                "email":      payload.email,
                "company":    payload.company,
                "use_case":   payload.use_case,
                "message":    payload.message,
            }).execute()
            supabase_ok = True
        except Exception as e:
            print(f"[Supabase] contact insert error: {e}")

    postgres_ok = _insert_contact_postgres(payload)

    if not supabase_ok and not postgres_ok:
        raise HTTPException(500, "Failed to save contact data")

    return {"status": "ok", "saved_supabase": supabase_ok, "saved_postgres": postgres_ok}


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /debug/routes
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/debug/routes")
async def debug_routes():
    return {"file": __file__, "routes": [r.path for r in app.router.routes]}
