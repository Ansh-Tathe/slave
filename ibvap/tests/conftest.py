"""
tests/conftest.py
Shared fixtures for IBVAP test suite.
"""
import sys
from pathlib import Path

# Make the repo root importable in all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
