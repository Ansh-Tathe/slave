"""
tests/test_p0_env.py
====================
P0 smoke tests — verify the environment is correctly set up.

Run:
    pytest tests/test_p0_env.py -v
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Python version
# =============================================================================
def test_python_version():
    """Python must be >= 3.11 for match-statement and typing improvements."""
    assert sys.version_info >= (3, 11), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} found; "
        "need >= 3.11"
    )


# =============================================================================
# 2. Key package imports
# =============================================================================
@pytest.mark.parametrize("package", [
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "loguru",
    "rich",
])
def test_package_importable(package: str):
    """Core packages must be importable (subset — torch tested separately)."""
    try:
        importlib.import_module(package)
    except ImportError as exc:
        pytest.fail(f"Cannot import '{package}': {exc}")


# =============================================================================
# 3. CUDA / PyTorch
# =============================================================================
def test_torch_importable():
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    assert hasattr(torch, "__version__")


def test_cuda_available():
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine — skipping GPU tests")


def test_fp16_tensor():
    """Verify FP16 tensors can be created and multiplied on GPU."""
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.ones(4, 4, dtype=torch.float16, device="cuda")
    b = torch.ones(4, 4, dtype=torch.float16, device="cuda")
    c = torch.matmul(a, b)
    assert c.shape == (4, 4)
    assert c.dtype == torch.float16


# =============================================================================
# 4. Repo directory structure
# =============================================================================
REQUIRED_DIRS = [
    "services/ingest",
    "services/detect_track",
    "services/analytics/fence",
    "services/analytics/anpr",
    "services/analytics/behaviour",
    "services/analytics/face",
    "services/analytics/night",
    "services/event_engine",
    "services/api",
    "models",
    "configs",
    "dashboard",
    "tests",
    "scripts",
    "docs",
]


@pytest.mark.parametrize("rel_dir", REQUIRED_DIRS)
def test_directory_exists(rel_dir: str):
    d = REPO_ROOT / rel_dir
    assert d.is_dir(), f"Expected directory missing: {d}"


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "docker-compose.yml",
    "configs/cameras.yaml",
    "configs/zones.yaml",
    "configs/rules.yaml",
    "scripts/check_gpu.py",
    "scripts/download_samples.py",
]


@pytest.mark.parametrize("rel_file", REQUIRED_FILES)
def test_file_exists(rel_file: str):
    f = REPO_ROOT / rel_file
    assert f.is_file(), f"Expected file missing: {f}"


# =============================================================================
# 5. Config YAML validity
# =============================================================================
def test_cameras_yaml_valid():
    import yaml
    config_path = REPO_ROOT / "configs" / "cameras.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    assert "cameras" in data, "cameras.yaml must have a 'cameras' key"
    assert isinstance(data["cameras"], list), "'cameras' must be a list"
    for cam in data["cameras"]:
        assert "id" in cam, f"Camera entry missing 'id': {cam}"
        assert "source" in cam, f"Camera {cam.get('id')} missing 'source'"


def test_zones_yaml_valid():
    import yaml
    config_path = REPO_ROOT / "configs" / "zones.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    assert "zones" in data, "zones.yaml must have a 'zones' key"


def test_rules_yaml_valid():
    import yaml
    config_path = REPO_ROOT / "configs" / "rules.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    assert "rules" in data, "rules.yaml must have a 'rules' key"
    assert "event_engine" in data, "rules.yaml must have an 'event_engine' key"
    for rule in data["rules"]:
        assert "id" in rule, f"Rule missing 'id': {rule}"
        assert "severity" in rule, f"Rule {rule.get('id')} missing 'severity'"


# =============================================================================
# 6. GPU check script runs without crashing
# =============================================================================
def test_check_gpu_script_runs():
    """
    check_gpu.py must exit 0 or 1 (failures are acceptable at this stage),
    but must NOT crash with an unhandled exception (exit 2+).
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_gpu.py"),
         "--skip-yolo"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), (
        f"check_gpu.py crashed (exit {result.returncode}):\n{result.stderr}"
    )
