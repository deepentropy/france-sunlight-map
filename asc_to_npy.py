#!/usr/bin/env python3
"""
Utility functions for converting RGEALTI ASC files to NumPy format.

ASC (ASCII Grid) format is a simple raster format with header metadata
followed by space-delimited elevation values.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import re


def read_asc_header(file_path: str) -> Dict[str, float]:
    """
    Read header information from ASC file.

    ASC file format header example:
        ncols         4000
        nrows         4000
        xllcorner     100000.0
        yllcorner     6200000.0
        cellsize      1.0
        NODATA_value  -99999

    Args:
        file_path: Path to ASC file

    Returns:
        Dictionary with header information
    """
    header = {}
    header_lines = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check if line is a header line (contains non-numeric key)
            if line and not line[0].isdigit() and line[0] != '-':
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].lower()
                    try:
                        value = float(parts[1])
                        header[key] = value
                        header_lines += 1
                    except ValueError:
                        break
            else:
                # We've reached the data section
                break

    header['header_lines'] = header_lines
    return header


def asc_to_npy(
    asc_path: str,
    npy_path: Optional[str] = None,
    keep_nodata: bool = False,
    nodata_replacement: Optional[float] = np.nan,
    dtype: str = 'float32'
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert ASC (ASCII Grid) file to NumPy array and save as NPY.

    Args:
        asc_path: Path to input ASC file
        npy_path: Path to output NPY file (optional, defaults to same name with .npy)
        keep_nodata: If True, keep NODATA values as-is; if False, replace them
        nodata_replacement: Value to replace NODATA with (default: np.nan)
        dtype: NumPy dtype for output array (default: 'float32')

    Returns:
        Tuple of (numpy_array, header_dict)

    Example:
        >>> data, header = asc_to_npy('elevation.asc', 'elevation.npy')
        >>> print(f"Shape: {data.shape}")
        >>> print(f"Cell size: {header['cellsize']} meters")
        >>> print(f"Valid data range: {np.nanmin(data)} to {np.nanmax(data)}")
    """
    # Read header
    header = read_asc_header(asc_path)

    # Get metadata
    ncols = int(header.get('ncols', 0))
    nrows = int(header.get('nrows', 0))
    nodata_value = header.get('nodata_value', -99999)
    header_lines = int(header.get('header_lines', 6))

    print(f"Reading ASC file: {asc_path}")
    print(f"  Dimensions: {nrows} rows x {ncols} cols")
    print(f"  Cell size: {header.get('cellsize', 'unknown')} meters")
    print(f"  NODATA value: {nodata_value}")

    # Read data, skipping header lines
    try:
        data = np.loadtxt(asc_path, skiprows=header_lines, dtype=dtype)
    except Exception as e:
        print(f"Error reading data with loadtxt: {e}")
        print("Trying alternative reading method...")

        # Alternative method: read line by line
        data_lines = []
        with open(asc_path, 'r') as f:
            # Skip header
            for _ in range(header_lines):
                next(f)

            # Read data
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    data_lines.append(values)

        data = np.array(data_lines, dtype=dtype)

    # Verify dimensions
    actual_rows, actual_cols = data.shape
    if actual_rows != nrows or actual_cols != ncols:
        print(f"Warning: Dimension mismatch!")
        print(f"  Expected: {nrows}x{ncols}")
        print(f"  Actual: {actual_rows}x{actual_cols}")

    # Handle NODATA values
    if not keep_nodata and nodata_replacement is not None:
        nodata_mask = data == nodata_value
        nodata_count = np.sum(nodata_mask)

        if nodata_count > 0:
            print(f"  Replacing {nodata_count} NODATA values with {nodata_replacement}")
            data[nodata_mask] = nodata_replacement

    # Print statistics
    if np.isnan(nodata_replacement):
        valid_data = data[~np.isnan(data)]
    else:
        valid_data = data[data != nodata_value]

    if len(valid_data) > 0:
        print(f"  Valid data range: {valid_data.min():.2f} to {valid_data.max():.2f}")
        print(f"  Valid data mean: {valid_data.mean():.2f}")
        print(f"  Valid data points: {len(valid_data)} ({len(valid_data)/data.size*100:.1f}%)")

    # Save to NPY if path provided
    if npy_path is None:
        npy_path = str(Path(asc_path).with_suffix('.npy'))

    np.save(npy_path, data)
    print(f"Saved to: {npy_path}")

    # Also save metadata
    metadata_path = str(Path(npy_path).with_suffix('.metadata.npy'))
    np.save(metadata_path, header)
    print(f"Saved metadata to: {metadata_path}")

    return data, header


def batch_convert_asc_to_npy(
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*.asc",
    **kwargs
) -> int:
    """
    Batch convert multiple ASC files to NPY format.

    Args:
        input_dir: Directory containing ASC files
        output_dir: Output directory (defaults to same as input_dir)
        pattern: Glob pattern for ASC files (default: "*.asc")
        **kwargs: Additional arguments to pass to asc_to_npy()

    Returns:
        Number of files converted

    Example:
        >>> count = batch_convert_asc_to_npy('RGEALTI/D001_5M/', pattern='*.asc')
        >>> print(f"Converted {count} files")
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all ASC files
    asc_files = sorted(input_path.glob(pattern))

    if not asc_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return 0

    print(f"Found {len(asc_files)} ASC file(s) to convert\n")

    converted = 0
    for asc_file in asc_files:
        try:
            npy_file = output_path / asc_file.with_suffix('.npy').name
            asc_to_npy(str(asc_file), str(npy_file), **kwargs)
            converted += 1
            print()  # Add spacing between files
        except Exception as e:
            print(f"Error converting {asc_file}: {e}\n")

    print(f"Successfully converted {converted}/{len(asc_files)} files")
    return converted


def load_npy_with_metadata(npy_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load NPY file along with its metadata.

    Args:
        npy_path: Path to NPY file

    Returns:
        Tuple of (numpy_array, header_dict)
    """
    data = np.load(npy_path)

    # Try to load metadata
    metadata_path = str(Path(npy_path).with_suffix('.metadata.npy'))
    try:
        metadata = np.load(metadata_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Warning: Metadata file not found: {metadata_path}")
        metadata = {}

    return data, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ASC (ASCII Grid) files to NumPy NPY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python3 asc_to_npy.py input.asc output.npy

  # Convert with NODATA as NaN
  python3 asc_to_npy.py input.asc --nodata-replacement nan

  # Convert with NODATA as 0
  python3 asc_to_npy.py input.asc --nodata-replacement 0

  # Batch convert all ASC files in directory
  python3 asc_to_npy.py --batch RGEALTI/D001_5M/

  # Batch convert to different directory
  python3 asc_to_npy.py --batch RGEALTI/D001_5M/ --output-dir processed/
        """
    )

    parser.add_argument(
        "input",
        help="Input ASC file or directory (for batch mode)"
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output NPY file (optional, defaults to same name)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all ASC files in input directory"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for batch mode"
    )

    parser.add_argument(
        "--pattern",
        default="*.asc",
        help="Glob pattern for batch mode (default: *.asc)"
    )

    parser.add_argument(
        "--keep-nodata",
        action="store_true",
        help="Keep NODATA values as-is"
    )

    parser.add_argument(
        "--nodata-replacement",
        default="nan",
        help="Value to replace NODATA with (default: nan). Use 'none' to keep as-is"
    )

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=['float16', 'float32', 'float64', 'int16', 'int32'],
        help="NumPy dtype for output (default: float32)"
    )

    args = parser.parse_args()

    # Parse nodata replacement
    if args.nodata_replacement.lower() == 'nan':
        nodata_replacement = np.nan
    elif args.nodata_replacement.lower() == 'none':
        nodata_replacement = None
    else:
        try:
            nodata_replacement = float(args.nodata_replacement)
        except ValueError:
            print(f"Error: Invalid nodata-replacement value: {args.nodata_replacement}")
            sys.exit(1)

    # Batch or single conversion
    if args.batch:
        batch_convert_asc_to_npy(
            args.input,
            args.output_dir or args.output,
            pattern=args.pattern,
            keep_nodata=args.keep_nodata,
            nodata_replacement=nodata_replacement,
            dtype=args.dtype
        )
    else:
        asc_to_npy(
            args.input,
            args.output,
            keep_nodata=args.keep_nodata,
            nodata_replacement=nodata_replacement,
            dtype=args.dtype
        )
