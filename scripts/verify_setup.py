#!/usr/bin/env python3
"""
Verify PointWSSIS installation and setup.

This script checks:
1. Python environment and dependencies
2. C++ extensions built correctly
3. DETECTRON2_DATASETS environment variable
4. Data structure and annotation files
5. CUDA availability

Usage:
    python verify_setup.py [--scenario SCENARIO]

Examples:
    python verify_setup.py              # Check basic setup
    python verify_setup.py --scenario 1p   # Verify files for 1% scenario
"""

import argparse
import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.7+)"
        )
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "detectron2",
        "adet",
        "cv2",
        "numpy",
        "pycocotools",
        "fvcore",
        "iopath",
        "portalocker",
    ]

    missing = []
    installed = []

    for package in required_packages:
        try:
            if package == "cv2":
                __import__("cv2")
            elif package == "adet":
                # Special check for adet to avoid circular import
                import importlib.util

                spec = importlib.util.find_spec("adet")
                if spec is None:
                    raise ImportError
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)

    for pkg in installed:
        print(f"  ✓ {pkg}")

    for pkg in missing:
        print(f"  ✗ {pkg} (not installed)")

    return len(missing) == 0


def check_cpp_extensions():
    """Check if C++ extensions are built."""
    print("\nChecking C++ extensions...")

    try:
        from detectron2 import _C as detectron2_C

        print("  ✓ detectron2._C")
    except ImportError as e:
        print(f"  ✗ detectron2._C (not built)")
        print(f"    Error: {e}")
        print("    Run: cd detectron2 && python -m pip install -e .")
        return False

    try:
        from adet import _C as adet_C

        print("  ✓ adet._C")
    except ImportError as e:
        print(f"  ✗ adet._C (not built)")
        print(f"    Error: {e}")
        print("    Run: cd AdelaiDet && python setup.py build develop --user")
        return False

    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_environment_variables():
    """Check required environment variables."""
    print("\nChecking environment variables...")

    data_root = os.getenv("DETECTRON2_DATASETS")
    if data_root:
        print(f"  ✓ DETECTRON2_DATASETS={data_root}")
        return data_root
    else:
        print("  ✗ DETECTRON2_DATASETS not set")
        print("    Run: export DETECTRON2_DATASETS=/path/to/data")
        return None


def check_data_structure(data_root, scenario=None):
    """Check data directory structure and files."""
    print("\nChecking data structure...")

    if not data_root:
        print("  ✗ Cannot check data structure (DETECTRON2_DATASETS not set)")
        return False

    data_root = Path(data_root)

    # Check main directories
    required_dirs = ["coco", "coco/train2017", "coco/val2017", "coco/annotations"]

    all_exist = True
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (not found)")
            all_exist = False

    # Check standard COCO annotations
    annotations_dir = data_root / "coco" / "annotations"
    standard_annotations = ["instances_train2017.json", "instances_val2017.json"]

    for filename in standard_annotations:
        filepath = annotations_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not found)")
            all_exist = False

    # Check scenario-specific annotations if requested
    if scenario:
        print(f"\nChecking {scenario} scenario annotations...")
        scenario_files = [
            f"instances_train2017_{scenario}_s.json",
            f"instances_train2017_{scenario}_w.json",
        ]

        for filename in scenario_files:
            filepath = annotations_dir / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (not found)")
                print(
                    f"    Download: python scripts/download_annotations.py {scenario.rstrip('p')}"
                )
                all_exist = False

    return all_exist


def check_pythonpath():
    """Check if required directories are in PYTHONPATH."""
    print("\nChecking PYTHONPATH...")

    pythonpath = os.getenv("PYTHONPATH", "")
    paths = pythonpath.split(os.pathsep)

    # When running scripts, these should be in PYTHONPATH
    suggested_paths = ["AdelaiDet", "detectron2", "MaskRefineNet"]

    found = []
    not_found = []

    for suggested in suggested_paths:
        if any(suggested in p for p in paths):
            found.append(suggested)
        else:
            not_found.append(suggested)

    if found:
        print("  ✓ Found in PYTHONPATH:")
        for p in found:
            print(f"    - {p}")

    if not_found:
        print("  ⚠ Not in PYTHONPATH (will be set by training scripts):")
        for p in not_found:
            print(f"    - {p}")
        print("    Note: The training scripts automatically set PYTHONPATH")

    return True  # This is not a critical check


def print_summary(checks):
    """Print summary of all checks."""
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(checks.values())
    total = len(checks)

    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")

    print("=" * 60)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All checks passed! Your setup is ready.")
        return True
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify PointWSSIS installation and setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        help="Check files for specific scenario (e.g., 1p, 5p, 10p)",
        default=None,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PointWSSIS Setup Verification")
    print("=" * 60)

    # Run all checks
    checks = {}

    checks["Python Version"] = check_python_version()
    checks["Dependencies"] = check_dependencies()
    checks["C++ Extensions"] = check_cpp_extensions()
    checks["CUDA"] = check_cuda()

    data_root = check_environment_variables()
    checks["Environment Variables"] = data_root is not None
    checks["Data Structure"] = check_data_structure(data_root, args.scenario)

    check_pythonpath()  # Informational only

    # Print summary
    success = print_summary(checks)

    if success:
        print("\nNext steps:")
        print("  1. Download annotations: python scripts/download_annotations.py 1 2 5")
        print("  2. Run training: bash scripts/coco_1p.sh")
        sys.exit(0)
    else:
        print("\nPlease fix the errors above before running training.")
        print("See SETUP_GUIDE.md for detailed instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
