# Local AI Crowd Monitoring System

A complete local AI crowd-monitoring system built with PyTorch, Ultralytics YOLO, ByteTrack, OpenCV, Streamlit, FastAPI, and PostgreSQL.

---

## System Architecture

```
crowd-monitoring-system/
├── app.py                      # Phase 2 & 3: Interactive Streamlit Monitoring Dashboard
├── api.py                      # Phase 3: FastAPI Backend Service & REST API
├── init_db.py                  # Phase 3: Database Schema Initialization & Seeding
├── benchmark_counting.py       # Phase 3: Ground-Truth Counting Benchmark CLI
├── train.py                    # Phase 1: YOLO Fine-Tuning CLI Entrypoint
├── evaluate.py                 # Phase 1: Test Split Evaluation CLI Entrypoint
├── export_yolov8.py            # Phase 1: Roboflow Dataset Export Helper
├── docker-compose.yml          # Phase 3: Local PostgreSQL Container Setup
├── .env.example                # Phase 3: Environment Variable Template (Placeholders only)
├── requirements.txt            # All System Dependencies
├── .gitignore                  # Excludes datasets, weights, runs, secrets, DB files
├── src/
│   ├── dataset_validator.py    # Pre-training dataset integrity & leakage checks
│   ├── trainer.py              # YOLO fine-tuning pipeline with augmentations
│   ├── evaluator.py            # Evaluation & metric reporting pipeline
│   ├── monitoring/
│       ├── tracker.py          # YOLO + ByteTrack tracking engine
│       ├── zone_manager.py     # Multi-zone density calculation, thresholds & timers
│       ├── heatmap.py          # Session Gaussian movement heatmap
│       └── video_stream.py     # Resilient Webcam, Video, RTSP & Demo ingestion
│   ├── storage/
│       ├── database.py         # SQLAlchemy engine with PostgreSQL & SQLite fallback
│       ├── models.py           # ORM models (zones, incidents, logs, detections)
│       └── repository.py       # Database CRUD operations
│   ├── notification/
│       └── email_notifier.py   # SMTP alerts, snapshot attachments & latency timer
│   └── analytics/
│       ├── system_metrics.py   # CPU, RAM, and GPU hardware utilization monitor
│       └── counting_evaluator.py # Ground-truth vs predicted counting accuracy engine
└── runs/
    ├── detect/
    │   ├── crowd_people_detector/  # Phase 1 trained weights (best.pt) & plots
    │   └── eval_test/              # Phase 1 held-out test evaluation curves
    └── incidents/                  # Phase 3 captured incident snapshot images
```

---

## 1. Setup & Environment Configuration

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (e.g. GeForce RTX 2050) or CPU
- PostgreSQL (via Docker Compose or local installation)

### Installation
Activate the virtual environment and install project dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Configure your credentials in `.env`:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_SENDER`, `SMTP_PASSWORD`, `ALERT_RECIPIENTS`
- `SMTP_ENABLED`: Set to `true` to dispatch real emails via SMTP, or keep `false` during development/testing.
- `INCIDENT_SNAPSHOT_DIR`: Output directory for captured incident images (default: `runs/incidents`).

---

## 2. PostgreSQL Storage Setup

### Option A: Using Docker Compose (Recommended if Docker is installed)
Start the PostgreSQL container:
```powershell
docker compose up -d
```

### Option B: Local PostgreSQL Service (If Docker is unavailable)
1. Install PostgreSQL 16+ on your system (e.g. using the Windows installer from `postgresql.org`).
2. Ensure the service is running:
   ```powershell
   Get-Service -Name *postgres*
   ```
3. Set your PostgreSQL password in `.env`.
4. *Resilient Fallback*: If PostgreSQL is temporarily unreachable or misconfigured, the database layer automatically logs a warning and falls back to a local SQLite database (`crowd_monitoring.db`) so operations never crash.

### Database Initialization
Run the initialization script to generate the database schema and seed default monitoring zones:
```powershell
python init_db.py
```

---

## 3. Launching the Interactive Dashboard

Launch the Streamlit monitoring dashboard:

```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Key Dashboard & Monitoring Features:
1. **Frame-Level Crowd Density & Incident Monitoring**: Real-time evaluation of crowd density based on total bounding boxes per frame (Low $\le 5$, Moderate $\le 15$, High $\le 25$, Critical $\ge 30$). Eliminates mandatory Zone A/B defaults; custom monitoring zones are now completely optional.
2. **Duplicate & Contained Box Suppression**: Built-in filter that eliminates duplicate bounding boxes where a single person has multiple detections (e.g., one box for the full person and a nested sub-box for their face or upper torso).
3. **Instant Video Hot-Swapping**: Uploading a new video immediately halts previous processing, flushes cache, clears heatmap/trackers, and streams the new video without requiring a browser refresh.
4. **Sustained Overcrowding Timers**: Real-time visual countdown timer until sustained overcrowding triggers an incident.
5. **Incident Snapshot Gallery & Alerts Table**: Tab displaying all logged incidents, location/zone names, timestamps, people count, snapshot images, and email delivery status.
6. **Hardware Utilization**: Real-time display showing CPU %, RAM %, and GPU VRAM % utilization.
7. **Interactive Counting Benchmark Tool**: Tab allowing on-the-fly accuracy benchmarking against ground-truth dataset frames.
8. **Phase 1 Held-Out Test Metrics**: Permanent display of verified Phase 1 test metrics (Precision 77.61%, Recall 73.45%, mAP50 72.60%, mAP50-95 40.52%).

---

## 4. Launching the FastAPI Backend Service

Start the FastAPI application with Uvicorn:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: Open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Primary API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System overview, database dialect, and weights status |
| `GET` | `/api/v1/status` | Current monitoring status, active zones, and CPU/GPU metrics |
| `POST`| `/api/v1/detect` | Run YOLO people detection on an uploaded image with duplicate suppression (`suppress_duplicates`, `containment_threshold`) |
| `GET` | `/api/v1/zones` | List all configured monitoring zones and threshold values |
| `POST`| `/api/v1/zones` | Create or update a zone configuration |
| `DELETE`| `/api/v1/zones/{id}` | Delete a monitoring zone |
| `GET` | `/api/v1/incidents` | Query incident history with snapshot image paths |
| `POST`| `/api/v1/incidents/{id}/resolve`| Mark an active incident as resolved |
| `GET` | `/api/v1/notifications`| Query notification delivery logs and latencies |
| `POST`| `/api/v1/alerts/test` | Trigger a simulated/live alert test with delivery latency measurement |
| `GET` | `/api/v1/metrics/test-metrics`| Retrieve verified Phase 1 held-out test evaluation results |
| `POST`| `/api/v1/metrics/ground-truth-compare`| Compare custom ground-truth counts against predictions |

---

## 5. Duplicate Box Suppression & Crowd Density Engine

### 1. Same-Person Duplicate Filter (Preserving Distinct Individuals)
In dense crowd scenes, standard YOLO detectors can output multiple overlapping boxes for the **same person**:
- One box covering the **full body** (head to feet).
- A second box covering only the **upper torso** or **face** of that exact same person.

Standard Non-Maximum Suppression (NMS) fails to remove these sub-boxes because $\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$ is below 0.30 when a face box is only ~15–25% of a full body box's area.

However, naive containment filters erroneously delete **distinct individuals** (such as people standing behind, seated persons, or children) whose bounding boxes happen to fall inside the rectangle of a larger foreground person.

**Our Anatomical Solution**: In [`src/monitoring/tracker.py`](file:///c:/Users/sarim/crowd%20monitoring%20system/src/monitoring/tracker.py), `is_same_person_duplicate` suppresses an inner box **only if it matches the anatomical signature of the same person**:
1. **Head Alignment (Face & Torso)**: For the same person, head top $|y_{1,\text{sub}} - y_{1,\text{full}}| \le 0.15 \cdot H_{\text{full}}$ (accounting for head tilts and hair).
   - **Upper Torso**: Covers upper 30% to 90% of body height, shoulder width ratio $\ge 0.40$, midline $|cx - cx| \le 0.25$.
   - **Face**: Ends at neck ($y_2 \le 38\% \cdot H$), aspect ratio $0.60 \le H/W \le 2.0$, and height $\le 35\%$.
2. **Ground & Feet Alignment**: Two boxes sharing the exact same ground footprint ($|y_2 - y_2| \le 10\% \cdot H$) and overlapping horizontally are identified as the same individual.
3. **Screen Boundary Duplicates**: Persons entering or exiting at frame edges ($x_1 \approx 0$ or $x_2 \approx \text{width}$) sharing the boundary are deduplicated.
4. **Moderate IoU Merging**: Box pairs with $\text{IoU} \ge 0.50$ (common for dual full-body proposals) are deduplicated.
5. **Distinct Persons are Preserved**: Children in front of adults, background people with lower heads, and seated persons do not match these criteria and are **strictly preserved**.

### 2. Frame-Level Crowd Density & Sustained Overcrowding
Crowd status is evaluated directly on the **total bounding boxes detected per frame**, without requiring arbitrary zones:
- **Low (Normal)**: Total count $\le$ Low threshold (default $\le 5$, Green)
- **Moderate**: Low $<$ Total count $\le$ Moderate threshold (default $6 - 15$, Amber)
- **High (Crowded)**: Moderate $<$ Total count $<$ Critical threshold (default $16 - 29$, Orange)
- **Critical (Overcrowded)**: Total count $\ge$ Critical threshold (default $\ge 30$, Red)

#### Sustained Overcrowding & Incident Creation:
- When total people count $\ge$ Critical threshold, a timer starts.
- If total people count remains $\ge$ Critical continuously for $\ge \text{sustained\_seconds}$ (default 3.0 seconds):
  1. An Incident is created: `INC-FRAME-<timestamp>`.
  2. The current annotated frame is captured to `runs/incidents/`.
  3. An SMTP email alert is dispatched with the snapshot attached (measuring delivery latency).
  4. The incident and notification log are saved in the database.
- **Incident Deduplication**: Only **one** incident is triggered during the same sustained overcrowding event.
- **Recurrence**: When total count drops below the critical threshold, the timer and incident flag reset, allowing a new incident if the crowd swells again.
- **Custom Zones**: Optional custom rectangular zones can still be added on top of frame-level monitoring if specific sub-areas need independent monitoring.

---

## 6. SMTP Email Notifications

### Email Configuration (e.g. Gmail)
To configure Gmail SMTP:
1. Enable 2-Step Verification in your Google Account.
2. Generate an **App Password** (16 characters) under Google Account -> Security -> App Passwords.
3. In `.env`:
   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USE_TLS=true
   SMTP_SENDER=your_email@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx
   ALERT_RECIPIENTS=admin@example.com,security@example.com
   SMTP_ENABLED=true
   ```
4. When `SMTP_ENABLED=false` (development default), alerts are safely simulated, delivery preparation latency is measured (~2.2 ms), and incidents are recorded without sending network emails.

---

## 7. Prototype Measurements & Benchmark Results

### 1. Ground-Truth Counting Benchmark
Run the counting accuracy benchmark against verified test dataset annotations:
```powershell
python benchmark_counting.py --samples 20 --device 0
```

#### Actual Collected Measurements (20 Test Frames, 226 Ground-Truth People):
- **Overall Counting Accuracy**: **81.42%**
- **Mean Absolute Error (MAE)**: **2.10 people/frame**
- **Root Mean Squared Error (RMSE)**: **3.02**
- **Mean Absolute Percentage Error (MAPE)**: **21.2%**

#### Mathematical Formulas
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\text{Pred}_i - \text{GT}_i|$$
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\text{Pred}_i - \text{GT}_i)^2}$$
$$\text{Overall Accuracy} = \max\left(0, 1 - \frac{\sum |\text{Pred}_i - \text{GT}_i|}{\sum \text{GT}_i}\right) \times 100\%$$

### 2. Phase 1 Held-Out Test Evaluation Metrics
Evaluated on 109 test images (585 people instances) on NVIDIA GeForce RTX 2050 GPU:
- **Precision (P)**: **77.61%**
- **Recall (R)**: **73.45%**
- **mAP @ 0.50**: **72.60%**
- **mAP @ 0.50:0.95**: **40.52%**
- **Detector-Only Latency**: **13.61 ms** (~73.5 FPS)
- **Full End-to-End Pipeline Speed (Streamlit + Detection + ByteTrack + Zones + Heatmap + UI)**: **~10.5 - 15.0 FPS**

### 3. Hardware Resource Usage Snapshot
- **CPU Utilization**: ~12.6%
- **System Memory (RAM)**: ~69.3% (11.1 GB used / 16.0 GB total)
- **GPU Utilization (NVIDIA RTX 2050)**: ~25 - 35% during live inference
- **GPU VRAM Allocation**: ~500 MB used / 4,096 MB total

---

## 8. Limitations & Scope

1. **Local Prototype**: Phase 3 is designed for local desktop and intranet deployment. It does not include multi-tenant user authentication, OAuth2/JWT auth gates, or distributed message queues (e.g. Celery/RabbitMQ).
2. **Ground-Plane Occlusion**: Extreme crowd density can cause brief tracking ID switches when individuals become heavily occluded; ByteTrack handles trajectory smoothing via Kalman filtering.
3. **SMTP Internet Requirement**: Dispatching real email alerts requires outbound SMTP network access (port 587/465); when offline, incidents are stored in PostgreSQL with `email_status="DISABLED"` or `"FAILED"`.
