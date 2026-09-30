# IBVAP — Intelligent Border Video Analytics Platform

> Software-only video analytics that turns existing IP CCTV cameras into an
> intelligent surveillance network. No proprietary smart-camera hardware required.

## Architecture

```
RTSP/File
   │
   ▼
[Ingest]  ──── frame queue ────►  [Detect & Track]
                                        │ tracks (JSON)
          ┌─────────────────────────────┤
          ▼         ▼         ▼         ▼
      [Fence]   [ANPR]  [Behaviour] [Face]
          └─────────┴─────────┴─────────┘
                          │ events (JSON)
                          ▼
                  [Event Engine]
               (rules / dedup / severity)
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
         [Storage]            [Alert API]
    (PostgreSQL + MinIO)   (REST + webhooks)
                                      │
                               [React Dashboard]
```

## Quick Start

```bash
pip install uv
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
python scripts/check_gpu.py
python scripts/download_samples.py
```

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| P0 | Env, repo skeleton, GPU check, sample videos | done |
| P1 | RTSP/file ingest + detection + tracking | done |
| P2 | Virtual fence, loitering, event engine | done |
| P3 | ANPR pipeline | done |
| P4 | Night-time enhancement | done |
| P5 | Suspicious-activity rules | done |
| P6 | Face detection + watchlist | done |
| P7 | FastAPI + dashboard + C2 webhooks | done |
| P8 | Benchmarks, hardening, docs | done |
| P9 | NL event search + incident reports | done |

## Directory Layout

```
ibvap/
├── services/
│   ├── ingest/          # RTSP/file reader, frame queue
│   ├── detect_track/    # YOLO + ByteTrack
│   ├── analytics/
│   │   ├── fence/       # virtual fence & tripwire
│   │   ├── anpr/        # plate detect + OCR
│   │   ├── behaviour/   # loitering, running, gathering
│   │   ├── face/        # detection + watchlist
│   │   └── night/       # low-light enhancement
│   ├── event_engine/    # rules, dedup, severity, clips
│   └── api/             # FastAPI, webhooks, RBAC
├── models/              # weights (git-ignored)
├── configs/
│   ├── cameras.yaml
│   ├── zones.yaml
│   └── rules.yaml
├── dashboard/           # React frontend
├── tests/               # pytest suite
├── scripts/             # check_gpu.py, download_samples.py
├── docs/
├── docker-compose.yml
└── requirements.txt
```
