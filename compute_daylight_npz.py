#!/usr/bin/env python3
"""
Example: Compute daylight hours from merged NPZ elevation data.

Handles files larger than RAM using memory mapping and chunk processing.

Usage:
    python3 compute_daylight_npz.py merged.npz daylight.npz
    python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 500
"""

import sys
import argparse
from compute import BatchProcessor
from datetime import datetime


def compute_daylight(input_npz, output_npz="daylight_results.npz",
                     batch_size=240, chunk_size=1000, pixel_size=5.0,
                     lat_center=46.0, lon_center=2.0):
    """
    Compute daylight hours from merged NPZ elevation file.

    Uses memory mapping to handle files larger than available RAM.

    Args:
        input_npz: Input NPZ file with merged elevation data
        output_npz: Output NPZ file for daylight results
        batch_size: Number of tiles to process in parallel on GPU
        chunk_size: Number of tiles to load into RAM at once
        pixel_size: Cell size in meters
        lat_center: Latitude of center point
        lon_center: Longitude of center point

    Returns:
        None (saves directly to file)
    """
    processor = BatchProcessor()

    processor.process_from_npz(
        input_npz=input_npz,
        output_npz=output_npz,
        date=datetime(2024, 6, 21),  # Summer solstice
        batch_size=batch_size,
        chunk_size=chunk_size,
        pixel_size=pixel_size,
        lat_center=lat_center,
        lon_center=lon_center
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute daylight hours from merged NPZ elevation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default chunk_size=1000 for 64GB RAM)
  python3 compute_daylight_npz.py merged.npz daylight.npz

  # Reduce chunk size for lower RAM (e.g., 32GB RAM)
  python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 500

  # Increase chunk size for more RAM (e.g., 128GB RAM)
  python3 compute_daylight_npz.py merged.npz daylight.npz --chunk-size 2000

  # Adjust batch size for GPU memory
  python3 compute_daylight_npz.py merged.npz daylight.npz --batch-size 100

Memory guidelines:
  - chunk_size=1000 tiles ≈ 15-20 GB RAM (suitable for 64GB total RAM)
  - chunk_size=500 tiles ≈ 8-10 GB RAM (suitable for 32GB total RAM)
  - chunk_size=2000 tiles ≈ 30-40 GB RAM (suitable for 128GB total RAM)

  Each tile is typically ~4000x4000 pixels × 4 bytes (float32) ≈ 64 MB
        """
    )

    parser.add_argument("input_npz", help="Input NPZ file with merged elevation data")
    parser.add_argument("output_npz", nargs="?", default="daylight_results.npz",
                       help="Output NPZ file for daylight results (default: daylight_results.npz)")
    parser.add_argument("--batch-size", type=int, default=240,
                       help="Number of tiles to process in parallel on GPU (default: 240)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Number of tiles to load into RAM at once (default: 1000)")
    parser.add_argument("--pixel-size", type=float, default=5.0,
                       help="Cell size in meters (default: 5.0 for 5M data)")
    parser.add_argument("--lat-center", type=float, default=46.0,
                       help="Latitude of center point (default: 46.0 for France)")
    parser.add_argument("--lon-center", type=float, default=2.0,
                       help="Longitude of center point (default: 2.0 for France)")

    args = parser.parse_args()

    compute_daylight(
        input_npz=args.input_npz,
        output_npz=args.output_npz,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        pixel_size=args.pixel_size,
        lat_center=args.lat_center,
        lon_center=args.lon_center
    )
