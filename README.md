# Katomaran Intelligent Face Tracker with Auto-Registration and Visitor Counting

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSES.md)
[![Testing: Pytest](https://img.shields.io/badge/testing-pytest-green.svg)](docs/testing.md)

An enterprise-grade, edge-ready computer vision application designed to process video files and live RTSP camera streams. The system performs real-time face detection, 512-dimensional biometric feature representation, multi-object temporal tracking, automated registration of previously unseen identities, re-identification across visits, track lifecycle event management (at most one ENTRY and EXIT per visit), cropped evidence image archiving, and persistent relational storage in SQLite.

---

## Table of Contents
1. [Objective & Key Features](#1-objective--key-features)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Prerequisites & System Requirements](#4-prerequisites--system-requirements)
5. [Windows Setup Instructions](#5-windows-setup-instructions)
6. [Linux Setup Instructions](#6-linux-setup-instructions)
7. [Model Acquisition & Weights Setup](#7-model-acquisition--weights-setup)
8. [Configuration Guide (config.json)](#8-configuration-guide-configjson)
9. [Running the Application](#9-running-the-application)
10. [RTSP Stream Configuration & Resiliency](#10-rtsp-stream-configuration--resiliency)
11. [Database Schema & SQL Queries](#11-database-schema--sql-queries)
12. [Track Lifecycle & Event State Machine](#12-track-lifecycle--event-state-machine)
13. [Sample Audit Logs (logs/events.log)](#13-sample-audit-logs-logseventslog)
14. [Automated Testing](#14-automated-testing)
15. [Performance Benchmarks & Profiling](#15-performance-benchmarks--profiling)
16. [Privacy, Security & Retention](#16-privacy-security--retention)
17. [Troubleshooting & Known Limitations](#17-troubleshooting--known-limitations)
18. [Documentation Links](#18-documentation-links)
19. [Demonstration Video Checklist & Placeholder](#19-demonstration-video-checklist--placeholder)

---

## 1. Objective & Key Features

### Core Objectives
1. **Face Detection**: Uses face-specific YOLO architectures (`yolov8n-face.pt`) for accurate face localization rather than body bounding boxes.
2. **Deep Facial Embeddings**: Employs InsightFace / ArcFace to extract 512-dimensional, L2-normalized feature vectors.
3. **Temporal Tracking**: Tracks multiple individuals across frames with motion velocity compensation and greedy IoU bipartite matching.
4. **Autonomous Registration**: Assigns sequential, persistent identifiers (`FACE_001`, `FACE_002`) to previously unseen individuals.
5. **Re-Identification Across Sessions**: When a visitor exits and returns later (even under a new tracker ID), their original Face ID is reused.
6. **Unique Visitor Counting**: Guarantees that unique visitor counts derive strictly from registered identities and do not increment on repeated frames or visits.
7. **Strict Lifecycle Events**: Emits at most **one ENTRY** and at most **one EXIT** event per physical presence session.
8. **Organized Image Archiving**: Stores face crops in date-partitioned folders (`logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`).
9. **Relational Persistence**: SQLite database (`database/visitors.db`) with Write-Ahead Logging (WAL) and foreign key constraints.
10. **Structured Audit Trail**: Continuous operational and biometric event logging to `logs/events.log`.

---

## 2. Architecture Overview

```mermaid
flowchart LR
    A[Video / RTSP Stream] --> B[VideoReader]
    B --> C[FaceDetector YOLO]
    C --> D[FaceTracker Motion/IoU]
    D --> E[FaceRecognizer ArcFace]
    E --> F[VisitorManager & Gallery]
    F --> G[EventManager State Machine]
    G --> H[(SQLite Database)]
    G --> I[ImageStorage logs/]
    G --> J[logs/events.log]
```

See the complete [Architecture Document](docs/architecture.md) for detailed topological sequence diagrams and data flow maps.

---

## 3. Technology Stack

- **Language**: Python 3.10
- **Computer Vision**: OpenCV (`opencv-python`), Pillow
- **Deep Learning**: PyTorch (`torch-cpu` / CUDA), ONNX Runtime
- **Face Detection**: Ultralytics YOLO Face (`yolov8n-face.pt`)
- **Face Recognition**: InsightFace (`buffalo_l` ArcFace 512-d)
- **Database**: SQLite 3 (WAL mode, parameterized transactions)
- **Performance Profiling**: `psutil`, custom moving-window telemetry
- **Testing**: `pytest`, `pytest-mock`
- **Optional Web Dashboard**: Streamlit

---

## 4. Prerequisites & System Requirements

- **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04 LTS)
- **Python**: Python 3.10 recommended for binary wheel compatibility.
- **Hardware (CPU Mode)**: Quad-core modern x86_64 CPU, 4 GB RAM, 2 GB free disk space.
- **Hardware (GPU Mode - Optional)**: NVIDIA GPU with CUDA 11.8/12.1+ and cuDNN.

---

## 5. Windows Setup Instructions

Run these commands in PowerShell from the project root:

```powershell
# 1. Create Python 3.10 virtual environment
py -3.10 -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade package managers
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies (CPU PyTorch + Vision packages)
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Run environment diagnostic check
python scripts/check_environment.py

# 6. Initialize database schema
python scripts/initialize_database.py
```

---

## 6. Linux Setup Instructions

```bash
# 1. Install system prerequisites
sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv libgl1-mesa-glx libglib2.0-0

# 2. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Initialize database
python scripts/initialize_database.py
```

---

## 7. Model Acquisition & Weights Setup

### Face Detector (`yolov8n-face.pt`)
The application includes an automated download helper. If downloading manually:
1. Download `yolov8n-face.pt` from official open-source releases:
   [GitHub Releases - yolov8-face](https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt)
2. Save weights into `models/face_detector.pt`:
   ```powershell
   Invoke-WebRequest -Uri "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt" -OutFile "models/face_detector.pt"
   ```

### Face Recognizer (`buffalo_l`)
InsightFace automatically downloads the `buffalo_l` model suite (containing `w600k_r50.onnx` ArcFace feature extractor) on first execution to `~/.insightface/models/buffalo_l`.

---

## 8. Configuration Guide (config.json)

Settings are validated on startup via `src/config.py`:

```json
{
  "video_source": "videos/sample_video.mp4",
  "input_type": "auto",
  "detection_skip_frames": 3,
  "exit_grace_frames": 30,
  "recognition_threshold": 0.45,
  "face_confidence_threshold": 0.50,
  "min_face_size": 30,
  "database_path": "database/visitors.db",
  "entry_image_dir": "logs/entries",
  "exit_image_dir": "logs/exits",
  "event_log_path": "logs/events.log",
  "yolo_model_path": "models/face_detector.pt",
  "recognition_model_name": "buffalo_l",
  "frame_resize_width": 960,
  "display_video": true,
  "save_detection_debug_video": false,
  "log_level": "INFO",
  "rtsp_reconnect_delay_sec": 2.0,
  "rtsp_max_reconnect_attempts": 5,
  "end_of_stream_policy": "mark_active"
}
```

### Parameter Reference
- `detection_skip_frames`: Detection runs once every `(detection_skip_frames + 1)` frames. For value `3`, frames 0, 4, 8 undergo full YOLO detection; intermediate frames are tracked using velocity motion prediction.
- `exit_grace_frames`: Consecutive missing frames before an absent face is confirmed to have exited.
- `recognition_threshold`: Cosine similarity cutoff (default `0.45`). Higher values (e.g. `0.55`) prevent false positives; lower values (e.g. `0.38`) tolerate harsh angles and lighting shifts.
- `end_of_stream_policy`: `'mark_active'` preserves in-scene presence; `'flush_exits'` forces EXIT event emission when video ends.

---

## 9. Running the Application (Windows PowerShell)

> [!IMPORTANT]
> Always activate the virtual environment first, or prefix every command with
> `.\venv\Scripts\python.exe` as shown below.

### Process Local Sample Video

```powershell
.\venv\Scripts\python.exe main.py --source videos/sample_video.mp4
```

### Process Local Video — Headless (No GUI Window)

Useful on servers, remote sessions, or when a display is unavailable:

```powershell
.\venv\Scripts\python.exe main.py --source videos/sample_video.mp4 --no-display
```

### Override Detection Interval and Recognition Threshold

```powershell
.\venv\Scripts\python.exe main.py --source videos/sample_video.mp4 --skip-frames 4 --threshold 0.50
```

### Connect to a Live RTSP Camera

> [!IMPORTANT]
> Replace `CAMERA_IP`, `PORT`, and `STREAM_PATH` with values supplied by your
> camera administrator.  **Never commit real credentials to source control.**

```powershell
.\venv\Scripts\python.exe main.py --source "rtsp://CAMERA_IP:PORT/STREAM_PATH"
```

With authentication credentials:

```powershell
.\venv\Scripts\python.exe main.py --source "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH"
```

Headless RTSP (no display window):

```powershell
.\venv\Scripts\python.exe main.py --source "rtsp://CAMERA_IP:554/STREAM_PATH" --no-display
```

### Stop the Stream Safely

- Press **`q`** or **`Esc`** in the OpenCV video window, **or**
- Press **`Ctrl+C`** in the terminal.

Both methods trigger the graceful shutdown path, flush remaining events, and
release the camera.

### View Saved Events and Face Images

```powershell
# Inspect the database (visitors + events table)
.\venv\Scripts\python.exe scripts/inspect_database.py --events-limit 50

# Browse face crops saved on ENTRY
explorer logs\entries

# Browse face crops saved on EXIT
explorer logs\exits

# Tail the structured audit log
Get-Content -Wait -Tail 40 logs\events.log
```

### Launch the Streamlit Dashboard

The dashboard reads from the SQLite database (`database/visitors.db`) and
displays all persisted visitors and events.  **Starting the dashboard does not
start the video processing pipeline** — the two processes are independent.

```powershell
.\venv\Scripts\streamlit.exe run dashboard.py
```

Then open `http://localhost:8501` in a browser.  The dashboard auto-refreshes
from the database; you can run `main.py` and the dashboard simultaneously.

---

## 10. RTSP Stream Configuration and Resiliency

### Obtaining Your Camera's RTSP URL

Ask your camera administrator for:
- **IP address or hostname** of the camera.
- **Port** (default is 554 for RTSP).
- **Stream path** (e.g. `/live`, `/h264/ch1/main/av_stream`, `/cam/realmonitor`).
- **Credentials** (username and password for the camera web interface).

Common URL formats by manufacturer:

| Manufacturer | Typical RTSP URL Pattern |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@IP/axis-media/media.amp` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Generic | `rtsp://user:pass@IP:554/live` |

Test the URL independently with VLC before running the application:

```
VLC → Media → Open Network Stream → paste your RTSP URL
```

### Checking Network Reachability

```powershell
# Ping the camera
ping CAMERA_IP

# Check whether RTSP port 554 is reachable
Test-NetConnection -ComputerName CAMERA_IP -Port 554
```

### Resiliency Features

| Feature | Detail |
|---|---|
| **Credential masking** | Passwords are replaced with `***:***` in all log messages and terminal output. |
| **TCP transport** | `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` eliminates packet-drop tearing. |
| **Startup retries** | The application retries up to `rtsp_max_reconnect_attempts` times (default 5) with `rtsp_reconnect_delay_sec`-second gaps before aborting. |
| **Mid-stream reconnection** | If the stream drops during processing, the same retry budget is applied automatically. |
| **Clean abort on failure** | If the camera is never reachable, the pipeline exits with a clear error message. No visitor records, events, or face images are created. |

### Interpreting Connection Errors

```
ERROR: Unable to connect to the RTSP camera.
       Source   : rtsp://***:***@192.168.1.100:554/live
       Attempts : 5 (all failed)
```

Possible causes:

1. Camera powered off or rebooting.
2. Wrong IP address or port number.
3. Incorrect credentials in the URL.
4. Firewall blocking port 554.
5. Camera stream requires a different path.

### Configuring RTSP Behaviour in `config.json`

```json
{
  "video_source": "rtsp://CAMERA_IP:554/STREAM_PATH",
  "rtsp_reconnect_delay_sec": 2.0,
  "rtsp_max_reconnect_attempts": 5,
  "detection_skip_frames": 4,
  "display_video": true
}
```

> [!CAUTION]
> Do **not** put real camera credentials in `config.json` if the file is
> committed to version control.  Use the `--source` CLI flag at runtime instead.



## 11. Database Schema & SQL Queries

### Schema Overview
- **`visitors`**: Persistent biometric identities.
  - `face_id` (TEXT, PK): E.g. `FACE_001`
  - `first_seen` (TEXT): First registration ISO timestamp
  - `last_seen` (TEXT): Latest detection ISO timestamp
  - `embedding` (BLOB): 512 float32 values (2048 bytes)
  - `total_visits` (INTEGER): Visit count
  - `status` (TEXT): `'active'`, `'exited'`, or `'active_at_stream_end'`
- **`events`**: Audit trail of confirmed entries and exits.
  - `event_id` (INTEGER, PK AUTOINCREMENT)
  - `face_id` (TEXT, FK -> `visitors.face_id`)
  - `event_type` (TEXT CHECK: `'ENTRY'` or `'EXIT'`)
  - `timestamp` (TEXT)
  - `image_path` (TEXT): Path to saved crop
  - `track_id` (INTEGER): Temporary track reference

### Inspecting Database via CLI
```powershell
python scripts/inspect_database.py --events-limit 25
```

### Example SQL Queries
```sql
-- 1. Accurate Unique Visitor Count
SELECT COUNT(*) AS unique_visitors FROM visitors;

-- 2. List all visitors with their total physical visits
SELECT face_id, first_seen, last_seen, total_visits, status FROM visitors;

-- 3. Audit trail for a specific visitor
SELECT event_id, event_type, timestamp, image_path, track_id 
FROM events 
WHERE face_id = 'FACE_001' 
ORDER BY event_id ASC;

-- 4. Count total entries and exits
SELECT event_type, COUNT(*) AS total_count FROM events GROUP BY event_type;
```

---

## 12. Track Lifecycle & Event State Machine

```mermaid
stateDiagram-v2
    [*] --> DETECTED: Face localized by YOLO
    DETECTED --> IDENTIFIED: ArcFace matched or registered
    IDENTIFIED --> ACTIVE: ENTRY event committed & crop saved
    ACTIVE --> EXIT_PENDING: Face missing < exit_grace_frames
    EXIT_PENDING --> ACTIVE: Face reappears within grace period
    EXIT_PENDING --> EXITED: Missing >= exit_grace_frames (EXIT event committed)
    EXITED --> [*]
```

---

## 13. Sample Audit Logs (logs/events.log)

```log
2026-10-17 16:45:01 [INFO] [katomaran] Starting Katomaran Intelligent Face Tracker application...
2026-10-17 16:45:01 [INFO] [katomaran.database] Initializing database schema at 'database/visitors.db'...
2026-10-17 16:45:02 [INFO] [katomaran.face_detector] Loading YOLO face detector from 'models/face_detector.pt'...
2026-10-17 16:45:03 [INFO] [katomaran.video_reader] Opening video source: videos/sample_video.mp4
2026-10-17 16:45:04 [INFO] [katomaran.event_manager] Track #1 initiated in state DETECTED.
2026-10-17 16:45:04 [INFO] [katomaran.visitor_manager] Auto-registered NEW visitor FACE_001. Total unique visitors now: 1.
2026-10-17 16:45:04 [INFO] [katomaran.event_manager] [EVENT: ENTRY] Confirmed for FACE_001 on track #1 (Event #1, Image: logs/entries/2026-10-17/FACE_001_20261017_164504_123456.jpg)
2026-10-17 16:45:25 [INFO] [katomaran.event_manager] [EVENT: EXIT] Confirmed for FACE_001 on track #1 after 30 missing frames. (Event #2)
```

---

## 14. Automated Testing

Run the full pytest suite (no real camera required — all RTSP tests use mocks):

```powershell
.\venv\Scripts\python.exe -m pytest -v tests/
```

Run with short tracebacks on failure:

```powershell
.\venv\Scripts\python.exe -m pytest -v --tb=short tests/
```

### Test Modules

| Test File | What It Covers |
|---|---|
| `tests/test_config.py` | Configuration loading, validation, path resolution, RTSP URL passthrough |
| `tests/test_database.py` | SQLite schema, CRUD, embedding serialisation |
| `tests/test_event_manager.py` | ENTRY/EXIT lifecycle state machine |
| `tests/test_image_storage.py` | Crop saving, boundary clipping, date partitioning |
| `tests/test_visitor_count.py` | Unique visitor counting accuracy |
| `tests/test_visitor_manager.py` | Cosine similarity, auto-registration, gallery persistence |
| `tests/test_video_reader.py` | RTSP URL detection, credential masking, startup failure, reconnection, file compatibility, resource cleanup |
| `tests/test_upload_processor.py` | Upload validation, filename sanitization, image detection pipeline, video subprocess, summary parsing |

> [!NOTE]
> `tests/test_video_reader.py` and `tests/test_upload_processor.py` use `unittest.mock` to simulate camera connections and subprocess execution.
> **No real RTSP camera is required to run these tests.**
> Live-camera validation remains pending until a valid RTSP stream is provided.



## 15. Performance Benchmarks & Profiling

Detailed CPU and GPU benchmarks are documented in [Performance Guide](docs/performance.md).

- **CPU Throughput (Ryzen / Core i7)**: ~32-38 FPS with `detection_skip_frames = 2`.
- **Memory Footprint**: ~450 MB working set.
- **Biometric Matching**: < 0.1 ms vector dot product against in-memory gallery.

---

## 16. Privacy, Security & Retention

- **Anonymous Identifiers**: The system generates anonymous IDs (`FACE_001`). No personally identifiable information (PII) or demographic classification is inferred.
- **Biometric Protection**: Embeddings and facial crops are stored locally and never transmitted to external cloud endpoints.
- **Data Retention**: Date-partitioned directories allow automated cron purging of image logs older than required retention policies (e.g. 30 days).

---

## 17. Troubleshooting & Known Limitations

1. **Camera Stream Black / Fails to Open**:
   - Verify network connectivity to RTSP camera IP.
   - Verify RTSP URL format and credentials.
2. **Missing Model Weights**:
   - Run `python scripts/check_environment.py` to confirm weights status.
3. **High CPU Usage**:
   - Increase `detection_skip_frames` from `3` to `4` or `5` in `config.json`.
   - Reduce `frame_resize_width` to `640`.

---

## 18. Documentation Links

- [AI Planning & Design Decisions](docs/planning.md)
- [System Architecture & Data Flows](docs/architecture.md)
- [Test Strategy & Verification Report](docs/testing.md)
- [Performance Benchmarking & Profiling](docs/performance.md)

---

## 19. Demonstration Video Checklist & Placeholder

- [ ] Video demonstration recording URL: `[DEMO VIDEO URL PENDING]`
- [ ] Technical Walkthrough Checklist:
  - [ ] Terminal launch of `main.py` with HUD overlay visible.
  - [ ] Demonstration of initial visitor registration (`FACE_001`).
  - [ ] Demonstration of ENTRY event crop saved in `logs/entries/`.
  - [ ] Demonstration of subject exit and EXIT event generation after grace period.
  - [ ] Demonstration of returning visitor re-identification (same `FACE_001` reused).
  - [ ] Demonstration that unique visitor count remains strictly accurate.
  - [ ] Database inspection using `python scripts/inspect_database.py`.

---

This project is a part of a hackathon run by https://katomaran.com
