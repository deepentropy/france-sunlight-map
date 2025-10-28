#!/usr/bin/env python3
"""
Merge multiple compressed NPZ files into a single compressed NPZ file.
"""

import numpy as np
from pathlib import Path
from typing import Union, List


def merge_npy_files(
    input_path: Union[str, List[str]] = "npy/",
    output_npy: str = "merged.npz",
    pattern: str = "*.npz"
) -> np.ndarray:
    """
    Merge all compressed NPZ files into a single compressed NPZ file.

    Args:
        input_path: Directory containing NPZ files or list of NPZ file paths (default: "npy/")
        output_npy: Output compressed NPZ file path (default: "merged.npz")
        pattern: Glob pattern for NPZ files (default: "*.npz")

    Returns:
        Merged numpy array
    """
    # Get list of NPZ files
    if isinstance(input_path, str):
        npy_files = sorted(Path(input_path).glob(pattern))
    else:
        npy_files = [Path(f) for f in input_path]

    if not npy_files:
        raise ValueError(f"No NPZ files found")

    # Load and concatenate all files
    arrays = []
    for npy_file in npy_files:
        with np.load(npy_file) as data:
            # Get first array from compressed file
            key = list(data.keys())[0]
            arrays.append(data[key])

    # Concatenate along first axis
    merged = np.concatenate(arrays, axis=0)

    # Save compressed
    np.savez_compressed(output_npy, merged)

    return merged
