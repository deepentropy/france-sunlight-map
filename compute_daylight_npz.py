#!/usr/bin/env python3
"""
Example: Compute daylight hours from merged NPZ elevation data.

Usage:
    python3 compute_daylight_npz.py merged.npz daylight.npz
"""

import sys
from compute import BatchProcessor
from datetime import datetime


def compute_daylight(input_npz, output_npz="daylight_results.npz",
                     batch_size=240, pixel_size=5.0,
                     lat_center=46.0, lon_center=2.0):
    """
    Compute daylight hours from merged NPZ elevation file.

    Args:
        input_npz: Input NPZ file with merged elevation data
        output_npz: Output NPZ file for daylight results
        batch_size: Number of tiles to process in parallel
        pixel_size: Cell size in meters
        lat_center: Latitude of center point
        lon_center: Longitude of center point

    Returns:
        Daylight array
    """
    processor = BatchProcessor()

    daylight = processor.process_from_npz(
        input_npz=input_npz,
        output_npz=output_npz,
        date=datetime(2024, 6, 21),  # Summer solstice
        batch_size=batch_size,
        pixel_size=pixel_size,
        lat_center=lat_center,
        lon_center=lon_center
    )

    return daylight


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 compute_daylight_npz.py <input.npz> [output.npz]")
        print("\nExample:")
        print("  python3 compute_daylight_npz.py merged.npz daylight.npz")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "daylight_results.npz"

    compute_daylight(input_file, output_file)
