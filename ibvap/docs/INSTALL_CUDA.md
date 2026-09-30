# CUDA / PyTorch Installation Guide (RTX 4050 / CUDA 12.x)

## Step 1 — Install CUDA Toolkit 12.1+

Download from NVIDIA:
  https://developer.nvidia.com/cuda-downloads

Choose:
  OS: Windows
  Architecture: x86_64
  Version: 12.1 (or 12.4)
  Type: exe (local)

After install, verify:
  nvcc --version
  nvidia-smi

## Step 2 — Install cuDNN

Download from: https://developer.nvidia.com/cudnn
Match your CUDA version. Extract DLLs to:
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\

## Step 3 — Create Python environment

  pip install uv
  uv venv .venv --python 3.11
  .venv\Scripts\activate

## Step 4 — Install PyTorch (CUDA 12.1 wheel)

  uv pip install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu121

Verify:
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Expected output:
  True  NVIDIA GeForce RTX 4050 Laptop GPU

## Step 5 — Install IBVAP requirements

  uv pip install -r requirements.txt

## Step 6 — (Optional) TensorRT 10.x

  pip install nvidia-tensorrt==10.0.1
  -- OR --
  Download TRT 10.x from https://developer.nvidia.com/tensorrt
  and follow the Python wheel install guide.

## Step 7 — Run GPU check

  python scripts/check_gpu.py

All checks should PASS. The YOLO warm-up will auto-download yolov8n.pt (~6 MB).

---

## RTX 4050 VRAM budget (6 GB)

| Component             | FP16 VRAM est. |
|-----------------------|---------------|
| YOLOv8n (1 stream)    | ~0.3 GB       |
| YOLOv8s (1 stream)    | ~0.5 GB       |
| ByteTrack state       | ~0.1 GB       |
| PaddleOCR (ANPR)      | ~0.4 GB       |
| SCRFD face det        | ~0.2 GB       |
| ArcFace embed         | ~0.1 GB       |
| FAISS GPU index       | ~0.2 GB       |
| OS + driver overhead  | ~0.5 GB       |
|                       |               |
| **Total (5 modules)** | **~2.3 GB**   |
| **Headroom**          | **~3.7 GB**   |

With FP16 + TensorRT, we can comfortably run 5-8 simultaneous 720p streams
at 10 FPS each on the RTX 4050. YOLOv8n is recommended for speed; upgrade to
YOLOv8s if precision on small objects (VisDrone) is insufficient.
