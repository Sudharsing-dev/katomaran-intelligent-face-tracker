"""Diagnostic script to verify runtime environment, hardware acceleration, and directory layout.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def check_python_version() -> bool:
    print("\n[1/6] Checking Python Runtime...")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"    Python Version: {ver_str} ({platform.architecture()[0]})")
    print(f"    Executable: {sys.executable}")
    if (v.major, v.minor) < (3, 8):
        print("    [FAIL] Python 3.8+ required.")
        return False
    elif (v.major, v.minor) > (3, 12):
        print(f"    [WARN] Python {ver_str} detected. Some prebuilt wheels (insightface/onnx) may need compilation.")
    else:
        print("    [PASS] Python version is well-supported.")
    return True


def check_disk_space() -> bool:
    print("\n[2/6] Checking Disk Space...")
    cwd = Path.cwd()
    total, used, free = shutil.disk_usage(cwd)
    free_gb = free / (1024 ** 3)
    print(f"    Current Drive: {cwd.anchor}")
    print(f"    Free Disk Space: {free_gb:.2f} GB")
    if free_gb < 2.0:
        print("    [WARN] Less than 2 GB free space available.")
        return False
    print("    [PASS] Sufficient disk space available.")
    return True


def check_acceleration() -> None:
    print("\n[3/6] Checking Hardware Acceleration...")
    # PyTorch
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"    PyTorch Version: {torch.__version__}")
        print(f"    CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"    GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print("    [INFO] Running on CPU inference mode.")
    except ImportError:
        print("    [INFO] PyTorch not yet installed in current environment.")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"    ONNX Runtime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"    Available Execution Providers: {', '.join(providers)}")
    except ImportError:
        print("    [INFO] ONNX Runtime not yet installed.")


def check_core_packages() -> bool:
    print("\n[4/6] Checking Core Computer Vision Packages...")
    packages = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("pytest", "pytest"),
        ("ultralytics", "ultralytics (YOLO)"),
        ("insightface", "insightface (ArcFace)"),
    ]
    all_ok = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [OK] {label}: {ver}")
        except ImportError:
            print(f"    [MISSING] {label}")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    print("\n[5/6] Checking Required Project Directories...")
    root = Path(__file__).resolve().parent.parent
    required = [
        root / "database",
        root / "models",
        root / "videos",
        root / "logs" / "entries",
        root / "logs" / "exits",
    ]
    for d in required:
        rel = d.relative_to(root)
        if d.is_dir():
            print(f"    [OK] Directory exists: {rel}")
        else:
            print(f"    [CREATING] Directory: {rel}")
            d.mkdir(parents=True, exist_ok=True)
    return True


def check_configuration() -> bool:
    print("\n[6/6] Checking Configuration File...")
    root = Path(__file__).resolve().parent.parent
    cfg = root / "config.json"
    if cfg.exists():
        print(f"    [OK] Found configuration at {cfg.name}")
        return True
    print(f"    [WARN] config.json missing at {root}. Creating from config.example.json...")
    example = root / "config.example.json"
    if example.exists():
        shutil.copy(example, cfg)
        print("    [OK] Created config.json from example template.")
        return True
    return False


def main() -> int:
    print("=" * 60)
    print("  Katomaran Face Tracker - Environment Diagnostic")
    print("=" * 60)
    py_ok = check_python_version()
    check_disk_space()
    check_acceleration()
    check_core_packages()
    check_directories()
    check_configuration()
    print("\n" + "=" * 60)
    print("  Diagnostic check completed.")
    print("=" * 60)
    return 0 if py_ok else 1


if __name__ == "__main__":
    sys.exit(main())
