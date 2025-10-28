#!/usr/bin/env python3
"""
Merge multiple NPY files into a single NPY file.
"""

import numpy as np
from pathlib import Path
from typing import Union, List


def merge_npy_files(
    input_path: Union[str, List[str]],
    output_npy: str,
    pattern: str = "*.npy"
) -> np.ndarray:
    """
    Merge all NPY files into a single NPY file.

    Args:
        input_path: Directory containing NPY files or list of NPY file paths
        output_npy: Output NPY file path
        pattern: Glob pattern for NPY files (default: "*.npy")

    Returns:
        Merged numpy array
    """
    # Get list of NPY files
    if isinstance(input_path, str):
        npy_files = sorted(Path(input_path).glob(pattern))
    else:
        npy_files = [Path(f) for f in input_path]

    if not npy_files:
        raise ValueError(f"No NPY files found")

    # Load and concatenate all files
    arrays = []
    for npy_file in npy_files:
        data = np.load(npy_file)
        arrays.append(data)

    # Concatenate along first axis
    merged = np.concatenate(arrays, axis=0)

    # Save
    np.save(output_npy, merged)

    return merged
