# IBVAP Performance Tuning & Hardware Capacity Guide

This guide provides engineering formulas, VRAM budgeting rules, and heuristic calibration tips for running IBVAP at scale.

---

## 1. Hardware Sizing & Stream Capacity Matrix

| Hardware Target | VRAM | Ingest Resolution | Max Concurrent 10 FPS Streams | Recommended Model Weights |
|---|---|---|---|---|
| **CPU Only (8-Core i7/Ryzen)** | N/A | 640x360 | 2–3 Streams | YOLOv8n (ONNX INT8 / CPU) |
| **NVIDIA RTX 4050 (Laptop)** | 6 GB | 1280x720 | 5–8 Streams | YOLOv8n (FP16 / TensorRT) |
| **NVIDIA RTX 3060 (12 GB)** | 12 GB | 1280x720 | 10–14 Streams | YOLOv8s (FP16) |
| **NVIDIA RTX 4090 (24 GB)** | 24 GB | 1920x1080 | 20–28 Streams | YOLOv8m (TensorRT FP16) |
| **NVIDIA A4000 (16 GB ECC)** | 16 GB | 1280x720 | 16–22 Streams | YOLOv8s (TensorRT FP16) |

---

## 2. VRAM Allocation Breakdown (RTX 4050 6 GB Example)

```
Total Available VRAM: 6.0 GB
├── PyTorch CUDA Context & OS Display Driver : ~0.6 GB
├── YOLOv8n (Batched FP16 Inference)         : ~0.8 GB
├── ByteTrack Association & Kalman Filter    : ~0.2 GB
├── PaddleOCR / ANPR Detector + Recognizer   : ~0.4 GB
├── SCRFD Face Det + ArcFace Embedding       : ~0.3 GB
├── FAISS Vector Similarity Index (GPU)      : ~0.2 GB
└── Headroom for Ingest Buffer & Bursts      : ~3.5 GB (Free)
```

---

## 3. Threshold Calibration for False Positive Suppression

### Object Detection (`configs/rules.yaml` -> `models.detect`)
- **Default Confidence (`conf_threshold: 0.40`):** Balances detection recall with precision.
- **Perimeter Security Optimization:** For outdoor fences with trees or shadows, raise `conf_threshold: 0.55` and enable temporal persistence in ByteTrack (`track_buffer: 30`). A detection must persist for at least 5 frames before raising an alarm.

### Virtual Fence Dwell Times (`configs/zones.yaml`)
- **Tripwire:** Use single-direction tripwires (`direction: inbound`) rather than bidirectional to suppress alarms on authorized personnel leaving a facility.
- **Loitering Dwell Time:** Set `dwell_time_s: 5.0` to `10.0` seconds to ignore workers briefly walking past a boundary.

### Face Recognition Watchlist
- **Cosine Similarity Threshold:**
  - `0.60–0.70`: High recall, potential false matches under extreme camera angles or low lighting.
  - `0.75+`: Standard security threshold for automated alert generation.
  - `0.82+`: Forensic grade matching (near zero false positive rate).
