# IBVAP Production Deployment Guide

This guide covers enterprise production deployments on **Linux (Ubuntu 22.04 LTS)** and **Windows Server**, including Docker containerization, systemd services, reverse proxy configuration, and RTSP stream rebroadcasting.

---

## 1. System Requirements

| Specification | Minimum (Lab / 2 Cameras) | Recommended Production (8–16 Cameras) |
|---|---|---|
| **CPU** | 4 Cores (Intel i5 / AMD Ryzen 5) | 16 Cores (Intel Xeon / AMD EPYC) |
| **RAM** | 16 GB DDR4 | 64 GB DDR4/DDR5 ECC |
| **GPU** | NVIDIA RTX 3060 / 4050 (6 GB) | NVIDIA RTX 4090 / A4000 (16–24 GB) |
| **Storage** | 256 GB NVMe SSD | 2 TB NVMe (OS/DB) + 16 TB SAS/NAS (Clips) |
| **OS** | Ubuntu 22.04 / Windows 11 | Ubuntu 22.04 Server / Windows Server 2022 |
| **Driver** | NVIDIA Driver 535+ / CUDA 12.x | NVIDIA Datacenter Driver 550+ |

---

## 2. Docker Compose Deployment (Recommended)

The easiest and most isolated way to run the entire IBVAP stack is via Docker Compose.

### Step 1: Install NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and configure passwords:
```bash
cp .env.example .env
```

### Step 3: Launch Stack
```bash
# Start storage, database, API, and analytics engine:
docker compose --profile full up -d
```

Verify service status:
```bash
docker compose ps
docker compose logs -f ibvap-api
```

---

## 3. Bare-Metal Linux Service (systemd)

For maximum GPU performance without container overhead, run IBVAP as a native systemd daemon.

Create `/etc/systemd/system/ibvap.service`:
```ini
[Unit]
Description=IBVAP Surveillance Analytics Service
After=network.target

[Service]
Type=simple
User=ibvap
WorkingDirectory=/opt/ibvap
ExecStart=/opt/ibvap/.venv/bin/python run_p7.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5s
Environment=PYTHONUNBUFFERED=1
Environment=IBVAP_SECRET_KEY=production-secret-key-32bytes-min

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ibvap
```

---

## 4. Windows Service Deployment (NSSM)

On Windows Server, use **NSSM (Non-Sucking Service Manager)**:

1. Download NSSM from [https://nssm.cc](https://nssm.cc) and copy `nssm.exe` to `C:\Windows\System32\`.
2. Open PowerShell as Administrator and run:
   ```powershell
   nssm install IBVAP "C:\Users\Administrator\ibvap\.venv\Scripts\python.exe" "run_p7.py --port 8000"
   nssm set IBVAP AppDirectory "C:\Users\Administrator\ibvap"
   nssm set IBVAP AppRestartDelay 5000
   nssm start IBVAP
   ```

---

## 5. Production Nginx Reverse Proxy (SSL + SSE)

When exposing IBVAP to operator networks, configure Nginx for SSL termination and Server-Sent Events (SSE) buffering bypass:

```nginx
server {
    listen 443 ssl http2;
    server_name surveillance.hq.internal;

    ssl_certificate     /etc/ssl/certs/ibvap.crt;
    ssl_certificate_key /etc/ssl/private/ibvap.key;

    # Dashboard & General API
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # CRITICAL: Disable proxy buffering for Real-Time SSE Streams
    location /api/v1/events/stream {
        proxy_pass http://127.0.0.1:8000/api/v1/events/stream;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
        chunked_transfer_encoding off;
    }
}
```
