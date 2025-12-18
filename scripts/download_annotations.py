#!/usr/bin/env python3
"""
Download PointWSSIS COCO annotation files from GitHub releases.

Usage:
    python download_annotations.py [percentages...]

Examples:
    python download_annotations.py 1 2 5      # Download 1%, 2%, 5% annotations
    python download_annotations.py all        # Download all percentages
    python download_annotations.py 1 --refined  # Include refined pseudo labels
"""

import argparse
import os
import sys
import requests
from pathlib import Path


BASE_URL = "https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/"

# All available percentage splits
ALL_PERCENTAGES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    print(f"Downloading: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
                print()  # New line after progress

        print(f"  ✓ Saved to: {dest_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading: {e}")
        return False


def download_annotations(
    percentages, annotations_dir, include_pseudo=False, include_refined=False
):
    """Download annotation files for specified percentages."""

    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for percent in percentages:
        print(f"\n{'=' * 60}")
        print(f"Processing {percent}% scenario")
        print("=" * 60)

        # Download strong (s) and weak (w) annotations
        for suffix in ["s", "w"]:
            filename = f"instances_train2017_{percent}p_{suffix}.json"
            url = BASE_URL + filename
            dest_path = annotations_dir / filename

            if dest_path.exists():
                print(f"Skipping {filename} (already exists)")
                continue

            if download_file(url, dest_path):
                success_count += 1
            else:
                fail_count += 1

        # Download pseudo labels if requested
        if include_pseudo:
            filename = f"instances_train2017_{percent}p_sw.json"
            url = BASE_URL + filename
            dest_path = annotations_dir / filename

            if not dest_path.exists():
                if download_file(url, dest_path):
                    success_count += 1
                else:
                    fail_count += 1

        # Download refined pseudo labels if requested
        if include_refined:
            filename = f"instances_train2017_{percent}p_sw_refined.json"
            url = BASE_URL + filename
            dest_path = annotations_dir / filename

            if not dest_path.exists():
                if download_file(url, dest_path):
                    success_count += 1
                else:
                    fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Download Summary:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print("=" * 60)

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download PointWSSIS COCO annotation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1 2 5           Download 1%%, 2%%, 5%% annotations
  %(prog)s all             Download all available annotations
  %(prog)s 1 --pseudo      Include pseudo labels (sw)
  %(prog)s 5 --refined     Include refined pseudo labels (sw_refined)
  %(prog)s all --pseudo --refined   Download everything
        """,
    )

    parser.add_argument(
        "percentages",
        nargs="+",
        help='Percentage splits to download (e.g., 1 2 5 10) or "all"',
    )
    parser.add_argument(
        "--pseudo", action="store_true", help="Also download pseudo labels (*_sw.json)"
    )
    parser.add_argument(
        "--refined",
        action="store_true",
        help="Also download refined pseudo labels (*_sw_refined.json)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: $DETECTRON2_DATASETS/coco/annotations)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        annotations_dir = args.output_dir
    else:
        data_root = os.getenv("DETECTRON2_DATASETS")
        if not data_root:
            print("Error: DETECTRON2_DATASETS environment variable not set")
            print("Please set it with: export DETECTRON2_DATASETS=/path/to/data")
            print("Or use --output-dir to specify the annotations directory")
            sys.exit(1)
        annotations_dir = os.path.join(data_root, "coco", "annotations")

    # Kaggle-specific fix: /kaggle/input is read-only
    if annotations_dir.startswith("/kaggle/input"):
        print(f"\n⚠️  Warning: Cannot write to /kaggle/input (read-only)")
        annotations_dir = "/kaggle/working/coco/annotations"
        print(f"   Redirecting to: {annotations_dir}")

    # Parse percentages
    if "all" in args.percentages:
        percentages = ALL_PERCENTAGES
        print(f"Downloading all available percentages: {percentages}")
    else:
        try:
            percentages = [int(p) for p in args.percentages]
            # Validate percentages
            invalid = [p for p in percentages if p not in ALL_PERCENTAGES]
            if invalid:
                print(f"Warning: Invalid percentages will be skipped: {invalid}")
                print(f"Available percentages: {ALL_PERCENTAGES}")
                percentages = [p for p in percentages if p in ALL_PERCENTAGES]
        except ValueError:
            print("Error: Percentages must be integers or 'all'")
            sys.exit(1)

    if not percentages:
        print("Error: No valid percentages specified")
        sys.exit(1)

    print(f"\nDownload Configuration:")
    print(f"  Output directory: {annotations_dir}")
    print(f"  Percentages: {percentages}")
    print(f"  Include pseudo labels: {args.pseudo}")
    print(f"  Include refined labels: {args.refined}")

    # Download annotations
    success = download_annotations(
        percentages=percentages,
        annotations_dir=annotations_dir,
        include_pseudo=args.pseudo,
        include_refined=args.refined,
    )

    if success:
        print("\n✓ All downloads completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some downloads failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
