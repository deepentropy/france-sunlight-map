#!/usr/bin/env python3
"""
Save all ASC files from a directory to a single NumPy file.
"""

import numpy as np
from pathlib import Path
from typing import Union, List

from tqdm import tqdm


def save_all_asc_to_single_npy(
    input_path: Union[str, List[str]],
    output_npy: str,
    nodata_value: float = -99999,
    nodata_replacement: float = np.nan
) -> np.ndarray:
    """
    Read all ASC files and save to a single NPY file.

    Args:
        input_path: Directory containing ASC files or list of ASC file paths
        output_npy: Output NPY file path
        nodata_value: NODATA value in ASC files (default: -99999)
        nodata_replacement: Value to replace NODATA with (default: np.nan)

    Returns:
        Stacked numpy array of shape (n_files, n_rows, n_cols)
    """
    # Get list of ASC files
    if isinstance(input_path, str):
        asc_files = sorted(Path(input_path).glob("*.asc"))
    else:
        asc_files = [Path(f) for f in input_path]

    if not asc_files:
        raise ValueError(f"No ASC files found in {input_path}")

    # Read all files
    arrays = []
    for asc_file in tqdm(asc_files):
        # Skip header lines (usually 6 lines)
        data = np.loadtxt(asc_file, skiprows=6, dtype='float32')

        # Replace NODATA
        data[data == nodata_value] = nodata_replacement

        arrays.append(data)

    # Stack all arrays
    stacked = np.stack(arrays, axis=0)

    # Save
    np.savez_compressed(output_npy, stacked)

    return stacked
