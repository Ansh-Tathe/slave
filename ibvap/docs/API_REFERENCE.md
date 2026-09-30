# IBVAP REST & Real-Time API Reference

Base URL: `http://localhost:8000/api/v1`  
Interactive OpenAPI UI: `http://localhost:8000/docs`  
ReDoc Reference: `http://localhost:8000/redoc`

---

## 1. Authentication & Security

Endpoints support two authentication mechanisms:
1. **JWT Bearer Token:** Header `Authorization: Bearer <token>`
2. **API Key (Machine-to-Machine / C2):** Header `X-API-Key: <key>`

### Login (`POST /auth/login`)
Exchange username and password for a JWT token:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "operator", "password": "operator123"}'
```
**Response:**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 86400,
  "role": "operator",
  "username": "operator"
}
```

---

## 2. Surveillance Events

### Ingest Event (`POST /events`)
```bash
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "TRIPWIRE_CROSS",
    "severity": "HIGH",
    "camera_id": "cam_01",
    "track_id": 42,
    "class_name": "person",
    "confidence": 0.94,
    "bbox": [100.0, 150.0, 200.0, 350.0],
    "center": [150.0, 250.0],
    "frame_id": 1240,
    "zone_id": "border_north",
    "metadata": {"direction": "inbound"}
  }'
```

### Query Events (`GET /events`)
Query parameters:
- `camera_id` (string): Filter by camera identifier.
- `severity` (string): `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`.
- `event_type` (string): e.g. `TRIPWIRE_CROSS`, `FACE_WATCHLIST_HIT`, `LOITERING`.
- `confirmed` (bool): `true` (confirmed), `false` (rejected), `null` (pending review).
- `limit` (int, default 50) & `offset` (int, default 0).

```bash
curl "http://localhost:8000/api/v1/events?camera_id=cam_01&severity=CRITICAL&limit=10"
```

### Real-Time Alert Stream (`GET /events/stream`)
Server-Sent Events (SSE) streaming live alerts:
```bash
curl -N http://localhost:8000/api/v1/events/stream
```
**Output stream:**
```text
: ibvap stream connected

data: {"event_id": "7820bb1b-...", "event_type": "TRIPWIRE_CROSS", "severity": "HIGH", "camera_id": "cam_01", ...}

: ping
```

### Operator Confirm/Reject Alert (`PATCH /events/{event_id}/confirm`)
```bash
curl -X PATCH http://localhost:8000/api/v1/events/7820bb1b-.../confirm \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"confirmed": true, "notes": "Verified intrusion by guard post"}'
```

---

## 3. Cameras & Health Telemetry

### List Cameras (`GET /cameras`)
```bash
curl http://localhost:8000/api/v1/cameras
```

### Camera Heartbeat (`POST /cameras/{camera_id}/heartbeat`)
```bash
curl -X POST http://localhost:8000/api/v1/cameras/cam_01/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"fps": 24.8, "frame_id": 18200, "status": "ONLINE"}'
```

### System Health (`GET /health`)
```bash
curl http://localhost:8000/health
```

### Telemetry Metrics (`GET /metrics`)
```bash
curl http://localhost:8000/metrics
```
