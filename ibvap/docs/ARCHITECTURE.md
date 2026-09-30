# IBVAP System Architecture & Dataflow Specification

The **Intelligent Border Video Analytics Platform (IBVAP)** is a modular, high-throughput surveillance engine designed to process multiple concurrent RTSP/IP camera feeds, extract semantic detections, track identities over time, evaluate complex spatial/behavioral rules, and dispatch verified alerts in real time.

```
                           RTSP / Video Streams
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Ingest / Watchdog  │
                         └──────────┬───────────┘
                                    │ Bounded Buffer (Drop-Oldest)
                                    ▼
                         ┌──────────────────────┐
                         │  Detection (YOLOv8)  │
                         └──────────┬───────────┘
                                    │ Detections [bbox, conf, class]
                                    ▼
                         ┌──────────────────────┐
                         │ Tracking (ByteTrack) │
                         └──────────┬───────────┘
                                    │ Tracks [track_id, bbox, history]
         ┌──────────────────────────┼─────────────────────────┐
         ▼                          ▼                         ▼
┌─────────────────┐       ┌───────────────────┐     ┌───────────────────┐
│ Virtual Fence   │       │ Behaviour Engine  │     │ Face / ANPR       │
│ (Polys/Lines)   │       │ (Loiter/Gather)   │     │ (Watchlist Match) │
└────────┬────────┘       └─────────┬─────────┘     └─────────┬─────────┘
         │                          │                         │
         └──────────────────────────┼─────────────────────────┘
                                    │ Raw Events
                                    ▼
                         ┌──────────────────────┐
                         │     Event Engine     │
                         │ (Dedup / Severity)   │
                         └──────────┬───────────┘
                                    │ Enriched Events
         ┌──────────────────────────┴─────────────────────────┐
         ▼                                                    ▼
┌────────────────────────┐                          ┌───────────────────┐
│ REST API & Live SSE    │                          │   C2 Webhook      │
│ (FastAPI + Dashboard)  │                          │   Dispatcher      │
└────────────────────────┘                          └───────────────────┘
```

---

## 1. Core Subsystems

### 1.1 Ingest & Resilient Stream Watchdog
- **Frame Reader (`services/ingest/reader.py`):** Uses OpenCV / PyAV to pull H.264/H.265 RTSP streams.
- **Stream Watchdog (`services/hardening/watchdog.py`):** Continuously monitors inter-frame arrival intervals. If packet loss or network drop exceeds `timeout_s` (default 5.0s), triggers automatic reconnection with exponential backoff and randomized jitter to avoid thundering-herd problems.
- **Bounded Buffer (`services/hardening/buffer.py`):** Employs a thread-safe ring buffer with a `DROP_OLDEST` policy (default 30 frames). If GPU inference momentarily lags, obsolete frames are discarded rather than accumulating unbounded latency.

### 1.2 Detection & Association Tracking
- **Object Detection (`services/detect_track/detector.py`):** YOLOv8/v11 optimized via PyTorch CUDA or TensorRT FP16 engines. Focuses on security-relevant classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7).
- **Multi-Object Tracking (`services/detect_track/tracker.py`):** ByteTrack implementation tracking bounding boxes and velocities across frames using Kalman filters.

### 1.3 Analytics Engines
1. **Virtual Fence (`services/analytics/fence/`):** Point-in-polygon ray casting and 2D line segment intersection algorithms operating in normalized `[0.0, 1.0]` coordinates. Supports dwell time thresholds and directional tripwires (inbound, outbound, bidirectional).
2. **Behaviour Analytics (`services/analytics/behaviour/`):**
   - *Loitering:* Tracks stationary time within a spatial radius.
   - *Speed & Running:* Computes rolling centroid displacement calibrated to pixels/meter.
   - *Gathering:* Connected-component spatial clustering measuring crowd formation.
   - *Abandoned Object:* Detects stationary backpacks, suitcases, or suspicious parcels left unattended.
3. **ANPR & Vehicle Identification (`services/analytics/anpr/`):** Plate localization combined with OCR and regex validation tailored for national license plate standards.
4. **Face Watchlist (`services/analytics/face/`):** SCRFD landmark detection, ArcFace 512-D embedding extraction, and sub-millisecond FAISS vector similarity search against enrolled watchlists.
5. **Night-time Enhancement (`services/analytics/night/`):** Automatic lux detection with adaptive CLAHE and Gamma illumination compensation.

### 1.4 Event Engine & Alert Deduplication
- **Deduplication:** State-based filter suppressing duplicate alert triggers for the same `(event_type, track_id, zone_id)` within a rolling window (`dedup_window_s`).
- **Snapshot Capture:** Automatically extracts track crops with configurable safety padding and compresses them to JPEG evidence artifacts.
- **Evidence Storage:** Emits structured JSON events to `data/logs/events.jsonl` and saves crops in `data/snapshots/`.

### 1.5 Command & Control API Layer
- **FastAPI Core (`services/api/`):** Asynchronous REST backend providing OpenAPI endpoints.
- **Server-Sent Events (SSE):** Push-based notification stream delivering sub-50ms alert broadcasts to operator dashboards.
- **C2 Webhooks:** Multi-target dispatcher delivering alert payloads to external Command & Control headquarters with retry queues.
