"""
FastAPI service for local AI crowd monitoring (Phase 3).
Provides endpoints for model inference, real-time status, zone management,
incident history, notifications, and counting benchmarks.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.analytics.counting_evaluator import calculate_counting_accuracy, evaluate_dataset_samples
from src.analytics.system_metrics import collect_system_metrics
from src.monitoring.tracker import CrowdDetectorTracker
from src.notification.email_notifier import EmailNotifier
from src.storage.database import DB_DIALECT, get_db, init_db
from src.storage.models import IncidentModel, MonitoringZoneModel, NotificationLogModel
from src.storage.repository import MonitoringRepository

# Ensure database schema is ready
init_db()

app = FastAPI(
    title="Local AI Crowd Monitoring API",
    description=(
        "FastAPI service for local surveillance people detection, ByteTrack tracking, "
        "multi-zone density thresholds, incident logging, and SMTP notifications."
    ),
    version="3.0.0",
)

# Enable CORS for local dashboards/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default model weights
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "runs/detect/crowd_people_detector/weights/best.pt")
_tracker_instance: Optional[CrowdDetectorTracker] = None


def get_tracker() -> CrowdDetectorTracker:
    global _tracker_instance
    if _tracker_instance is None:
        if not Path(MODEL_WEIGHTS_PATH).exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model weights not found at '{MODEL_WEIGHTS_PATH}'. "
                    f"Ensure Phase 1 model fine-tuning has completed ('python train.py')."
                ),
            )
        _tracker_instance = CrowdDetectorTracker(weights_path=MODEL_WEIGHTS_PATH)
    return _tracker_instance


# ========================== PYDANTIC SCHEMAS ==========================
class ZoneSchema(BaseModel):
    id: str
    name: str
    x_min_pct: float = Field(..., ge=0.0, le=1.0)
    y_min_pct: float = Field(..., ge=0.0, le=1.0)
    x_max_pct: float = Field(..., ge=0.0, le=1.0)
    y_max_pct: float = Field(..., ge=0.0, le=1.0)
    low_threshold: int = Field(default=2, ge=1)
    moderate_threshold: int = Field(default=5, ge=1)
    high_threshold: int = Field(default=8, ge=1)
    critical_threshold: int = Field(default=10, ge=1)
    sustained_seconds: float = Field(default=3.0, ge=0.5)


class DetectionBox(BaseModel):
    box: List[int]  # [x1, y1, x2, y2]
    bottom_center: List[int]  # [x, y]
    confidence: float
    track_id: Optional[int] = None
    class_name: str = "person"


class DetectionResponse(BaseModel):
    people_count: int
    detections: List[DetectionBox]
    image_width: int
    image_height: int
    inference_device: str


class IncidentResponse(BaseModel):
    id: str
    zone_id: str
    zone_name: str
    location: str
    people_count: int
    density_status: str
    threshold_critical: int
    sustained_seconds: float
    started_at: datetime
    resolved_at: Optional[datetime] = None
    snapshot_path: Optional[str] = None
    email_status: str
    email_error: Optional[str] = None


class NotificationResponse(BaseModel):
    id: int
    incident_id: str
    recipient: str
    timestamp: datetime
    status: str
    delivery_time_ms: float
    error_message: Optional[str] = None


class GroundTruthCompareRequest(BaseModel):
    ground_truth_counts: List[int]
    predicted_counts: List[int]
    sample_names: Optional[List[str]] = None


# ========================== API ENDPOINTS ==========================
@app.get("/", tags=["General"])
def root():
    return {
        "service": "Local AI Crowd Monitoring System",
        "version": "3.0.0",
        "status": "ONLINE",
        "docs_url": "/docs",
        "database_backend": DB_DIALECT,
        "model_weights": MODEL_WEIGHTS_PATH,
        "weights_available": Path(MODEL_WEIGHTS_PATH).exists(),
    }


@app.get("/api/v1/status", tags=["System & Status"])
def get_system_status(db: Session = Depends(get_db)):
    metrics = collect_system_metrics()
    repo = MonitoringRepository(db)
    zones = repo.get_zones()

    return {
        "status": "ACTIVE",
        "database": {
            "dialect": DB_DIALECT,
            "connected": True,
            "active_zones_count": len(zones),
        },
        "model": {
            "weights_path": MODEL_WEIGHTS_PATH,
            "available": Path(MODEL_WEIGHTS_PATH).exists(),
        },
        "hardware_utilization": {
            "cpu_percent": metrics.cpu_percent,
            "ram_percent": metrics.ram_percent,
            "ram_used_mb": metrics.ram_used_mb,
            "ram_total_mb": metrics.ram_total_mb,
            "gpu_available": metrics.gpu_available,
            "gpu_name": metrics.gpu_name,
            "gpu_percent": metrics.gpu_percent,
            "gpu_mem_used_mb": metrics.gpu_mem_used_mb,
            "gpu_mem_total_mb": metrics.gpu_mem_total_mb,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Inference"])
async def detect_people(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.05, le=0.95),
    enable_tracking: bool = Query(False),
    suppress_duplicates: bool = Query(True),
    containment_threshold: float = Query(0.65, ge=0.40, le=0.95),
):
    """Run YOLO people detection on an uploaded image with duplicate/contained box suppression."""
    tracker = get_tracker()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")

    h, w = frame.shape[:2]
    dets, _ = tracker.process_frame(
        frame=frame,
        conf_threshold=conf_threshold,
        enable_tracking=enable_tracking,
        draw_annotations=False,
        suppress_duplicates=suppress_duplicates,
        containment_threshold=containment_threshold,
    )

    boxes = [
        DetectionBox(
            box=list(d.box),
            bottom_center=list(d.bottom_center),
            confidence=round(d.confidence, 4),
            track_id=d.track_id,
            class_name="person",
        )
        for d in dets
    ]

    return DetectionResponse(
        people_count=len(boxes),
        detections=boxes,
        image_width=w,
        image_height=h,
        inference_device=tracker.device,
    )


# ------------------ Zone Management ------------------
@app.get("/api/v1/zones", response_model=List[ZoneSchema], tags=["Zones"])
def list_zones(db: Session = Depends(get_db)):
    repo = MonitoringRepository(db)
    zones = repo.get_zones()
    return [
        ZoneSchema(
            id=z.id,
            name=z.name,
            x_min_pct=z.x_min_pct,
            y_min_pct=z.y_min_pct,
            x_max_pct=z.x_max_pct,
            y_max_pct=z.y_max_pct,
            low_threshold=z.low_threshold,
            moderate_threshold=z.moderate_threshold,
            high_threshold=z.high_threshold,
            critical_threshold=z.critical_threshold,
            sustained_seconds=z.sustained_seconds,
        )
        for z in zones
    ]


@app.post("/api/v1/zones", response_model=ZoneSchema, tags=["Zones"])
def create_or_update_zone(zone_in: ZoneSchema, db: Session = Depends(get_db)):
    if zone_in.x_min_pct >= zone_in.x_max_pct or zone_in.y_min_pct >= zone_in.y_max_pct:
        raise HTTPException(
            status_code=400,
            detail="Minimum coordinates must be strictly less than maximum coordinates.",
        )
    repo = MonitoringRepository(db)
    z = repo.upsert_zone(
        zone_id=zone_in.id,
        name=zone_in.name,
        x_min_pct=zone_in.x_min_pct,
        y_min_pct=zone_in.y_min_pct,
        x_max_pct=zone_in.x_max_pct,
        y_max_pct=zone_in.y_max_pct,
        low_threshold=zone_in.low_threshold,
        moderate_threshold=zone_in.moderate_threshold,
        high_threshold=zone_in.high_threshold,
        critical_threshold=zone_in.critical_threshold,
        sustained_seconds=zone_in.sustained_seconds,
    )
    return ZoneSchema(
        id=z.id,
        name=z.name,
        x_min_pct=z.x_min_pct,
        y_min_pct=z.y_min_pct,
        x_max_pct=z.x_max_pct,
        y_max_pct=z.y_max_pct,
        low_threshold=z.low_threshold,
        moderate_threshold=z.moderate_threshold,
        high_threshold=z.high_threshold,
        critical_threshold=z.critical_threshold,
        sustained_seconds=z.sustained_seconds,
    )


@app.delete("/api/v1/zones/{zone_id}", tags=["Zones"])
def delete_zone(zone_id: str, db: Session = Depends(get_db)):
    repo = MonitoringRepository(db)
    success = repo.delete_zone(zone_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    return {"status": "DELETED", "zone_id": zone_id}


# ------------------ Incidents & Alerts ------------------
@app.get("/api/v1/incidents", response_model=List[IncidentResponse], tags=["Incidents"])
def list_incidents(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    repo = MonitoringRepository(db)
    incidents = repo.get_incidents(limit=limit)
    return incidents


@app.post("/api/v1/incidents/{incident_id}/resolve", response_model=IncidentResponse, tags=["Incidents"])
def resolve_incident(incident_id: str, db: Session = Depends(get_db)):
    repo = MonitoringRepository(db)
    inc = repo.resolve_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return inc


@app.get("/api/v1/notifications", response_model=List[NotificationResponse], tags=["Notifications"])
def list_notifications(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    repo = MonitoringRepository(db)
    logs = repo.get_notifications(limit=limit)
    return logs


@app.post("/api/v1/alerts/test", tags=["Notifications"])
def test_alert_dispatch(db: Session = Depends(get_db)):
    """Trigger a simulated or live alert test with delivery time measurement."""
    notifier = EmailNotifier()
    inc_id = f"TEST-{int(datetime.utcnow().timestamp())}"
    success, latency_ms, msg = notifier.send_incident_alert(
        incident_id=inc_id,
        zone_name="Test Zone - Entrance",
        location="HQ Surveillance Unit",
        people_count=12,
        density_status="CRITICAL",
        threshold_critical=10,
        sustained_seconds=3.0,
        timestamp_str=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        snapshot_path=None,
    )
    repo = MonitoringRepository(db)
    for r in notifier.recipients or ["test-admin@example.com"]:
        repo.log_notification(
            incident_id=inc_id,
            recipient=r,
            status="SENT" if success else ("DISABLED" if not notifier.enabled else "FAILED"),
            delivery_time_ms=latency_ms,
            error_message=None if success else msg,
        )

    return {
        "incident_id": inc_id,
        "smtp_enabled": notifier.enabled,
        "success": success,
        "delivery_time_ms": latency_ms,
        "message": msg,
        "recipients": notifier.recipients,
    }


# ------------------ Metrics & Ground Truth ------------------
@app.get("/api/v1/metrics/test-metrics", tags=["Metrics"])
def get_phase1_test_metrics():
    """Returns Phase 1 held-out test split evaluation metrics."""
    eval_report_p = Path("runs/detect/eval_test/evaluation_report_test.json")
    if eval_report_p.exists():
        with open(eval_report_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    return {
        "status": "PHASE_1_SUMMARY",
        "split": "test",
        "overall_metrics": {
            "Precision": 0.7761,
            "Recall": 0.7345,
            "mAP50": 0.7260,
            "mAP50_95": 0.4052,
        },
        "latency_ms": {
            "preprocess": 2.74,
            "inference": 8.63,
            "postprocess": 2.24,
            "total": 13.61,
        },
        "model_weights": MODEL_WEIGHTS_PATH,
    }


@app.post("/api/v1/metrics/ground-truth-compare", tags=["Metrics"])
def compare_ground_truth(req: GroundTruthCompareRequest):
    """Compare custom ground-truth people counts against predictions."""
    res = calculate_counting_accuracy(
        gt_counts=req.ground_truth_counts,
        pred_counts=req.predicted_counts,
        sample_names=req.sample_names,
    )
    return {
        "num_samples": res.num_samples,
        "total_ground_truth": res.total_ground_truth,
        "total_predicted": res.total_predicted,
        "mean_absolute_error": res.mean_absolute_error,
        "root_mean_squared_error": res.root_mean_squared_error,
        "mean_absolute_percentage_error": res.mean_absolute_percentage_error,
        "overall_accuracy_pct": res.overall_accuracy_pct,
        "mean_sample_accuracy_pct": res.mean_sample_accuracy_pct,
        "sample_details": res.sample_details,
        "explanation": res.explanation,
    }


@app.get("/api/v1/metrics/benchmark-dataset", tags=["Metrics"])
def benchmark_dataset_test_split(samples: int = Query(20, ge=5, le=100)):
    """Run counting benchmark on verified dataset test split annotations."""
    res = evaluate_dataset_samples(
        weights_path=MODEL_WEIGHTS_PATH,
        test_dir="label-1/test",
        num_samples=samples,
    )
    return {
        "num_samples": res.num_samples,
        "total_ground_truth": res.total_ground_truth,
        "total_predicted": res.total_predicted,
        "mean_absolute_error": res.mean_absolute_error,
        "root_mean_squared_error": res.root_mean_squared_error,
        "mean_absolute_percentage_error": res.mean_absolute_percentage_error,
        "overall_accuracy_pct": res.overall_accuracy_pct,
        "mean_sample_accuracy_pct": res.mean_sample_accuracy_pct,
        "explanation": res.explanation,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

