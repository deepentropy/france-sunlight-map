#!/usr/bin/env python3
"""
Example usage of ASC to NPY conversion functions.
"""

from asc_to_npy import asc_to_npy, load_npy_with_metadata, batch_convert_asc_to_npy
import numpy as np


def example_single_conversion():
    """Example: Convert a single ASC file to NPY."""
    print("="*60)
    print("Example 1: Single file conversion")
    print("="*60)

    # Convert single file with NODATA as NaN
    data, header = asc_to_npy(
        'elevation.asc',
        'elevation.npy',
        nodata_replacement=np.nan,
        dtype='float32'
    )

    print(f"\nData shape: {data.shape}")
    print(f"Cell size: {header.get('cellsize', 'unknown')} meters")
    print(f"Corner coordinates: ({header.get('xllcorner')}, {header.get('yllcorner')})")


def example_batch_conversion():
    """Example: Batch convert all ASC files in a directory."""
    print("\n" + "="*60)
    print("Example 2: Batch conversion")
    print("="*60)

    # Convert all ASC files in directory
    count = batch_convert_asc_to_npy(
        input_dir='RGEALTI/D001_5M/',
        output_dir='RGEALTI/D001_5M/npy/',
        pattern='*.asc',
        nodata_replacement=np.nan,
        dtype='float32'
    )

    print(f"Converted {count} files")


def example_load_and_analyze():
    """Example: Load NPY file and analyze the data."""
    print("\n" + "="*60)
    print("Example 3: Load and analyze")
    print("="*60)

    # Load the converted data
    data, metadata = load_npy_with_metadata('elevation.npy')

    # Analyze the data
    valid_data = data[~np.isnan(data)]

    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Memory size: {data.nbytes / 1024 / 1024:.2f} MB")
    print(f"\nStatistics (valid data only):")
    print(f"  Min elevation: {valid_data.min():.2f} m")
    print(f"  Max elevation: {valid_data.max():.2f} m")
    print(f"  Mean elevation: {valid_data.mean():.2f} m")
    print(f"  Std deviation: {valid_data.std():.2f} m")
    print(f"  Valid points: {len(valid_data)} ({len(valid_data)/data.size*100:.1f}%)")


def example_custom_processing():
    """Example: Convert with custom NODATA handling."""
    print("\n" + "="*60)
    print("Example 4: Custom NODATA handling")
    print("="*60)

    # Replace NODATA with 0 instead of NaN
    data, header = asc_to_npy(
        'elevation.asc',
        'elevation_zero_nodata.npy',
        nodata_replacement=0.0,
        dtype='float32'
    )

    print(f"NODATA values replaced with 0.0")
    print(f"Min value: {data.min():.2f}")
    print(f"Max value: {data.max():.2f}")


if __name__ == "__main__":
    print("ASC to NPY Conversion Examples")
    print("\nThese examples demonstrate different use cases.")
    print("Modify the file paths to match your data.")
    print("\n" + "="*60)

    # Uncomment the examples you want to run:

    # example_single_conversion()
    # example_batch_conversion()
    # example_load_and_analyze()
    # example_custom_processing()

    print("\nTo use these functions in your code:")
    print("""
    from asc_to_npy import asc_to_npy
    import numpy as np

    # Convert ASC to NPY
    data, header = asc_to_npy('input.asc', 'output.npy', nodata_replacement=np.nan)

    # Access data
    print(f"Elevation at [100, 100]: {data[100, 100]:.2f} m")

    # Get valid data statistics
    valid_data = data[~np.isnan(data)]
    print(f"Mean elevation: {valid_data.mean():.2f} m")
    """)
