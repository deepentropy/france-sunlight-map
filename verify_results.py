#!/usr/bin/env python3
"""
Verify and analyze daylight computation results.

Checks for:
- Valid data ranges
- Edge continuity between neighbors
- Highest/lowest daylight points
- Google Maps coordinates for verification
"""

import numpy as np
from pathlib import Path
import pickle
from pyproj import Transformer
import json
import sys

def load_spatial_index(index_file="asc_spatial_index.pkl"):
    """Load spatial index."""
    with open(index_file, 'rb') as f:
        return pickle.load(f)


def get_tile_coordinates(tile_info, row, col):
    """
    Convert pixel position to geographic coordinates.

    Args:
        tile_info: Tile metadata from spatial index
        row: Row index in array
        col: Column index in array

    Returns:
        (x_lambert93, y_lambert93) coordinates
    """
    header = tile_info['header']

    # Calculate coordinates
    x = header['xllcorner'] + col * header['cellsize']
    y = header['yllcorner'] + (header['nrows'] - row) * header['cellsize']

    return x, y


def lambert93_to_latlon(x, y):
    """Convert Lambert 93 to WGS84 lat/lon."""
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def analyze_tile_results(
    results_dir="daylight_results_tiles",
    index_file="asc_spatial_index.pkl",
    num_tiles=None
):
    """
    Analyze daylight computation results.

    Args:
        results_dir: Directory with result files
        index_file: Spatial index file
        num_tiles: Number of tiles to analyze (None = all)
    """
    print("="*70)
    print("DAYLIGHT RESULTS VERIFICATION")
    print("="*70)

    # Load spatial index
    print(f"\nLoading spatial index from {index_file}...")
    spatial_index = load_spatial_index(index_file)

    # Find result files
    results_path = Path(results_dir)
    result_files = sorted(results_path.glob("*_daylight.npy"))

    if not result_files:
        print(f"ERROR: No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files")

    if num_tiles:
        result_files = result_files[:num_tiles]
        print(f"Analyzing first {num_tiles} files")

    print("\n" + "="*70)
    print("TILE-BY-TILE ANALYSIS")
    print("="*70)

    all_stats = []
    highest_points = []

    for i, result_file in enumerate(result_files):
        print(f"\n[{i+1}/{len(result_files)}] {result_file.name}")

        # Load data
        daylight = np.load(result_file)

        # Get tile info
        tile_name = result_file.stem.replace('_daylight', '')

        # Find corresponding tile in index
        tile_id = None
        for tid, info in spatial_index['tiles'].items():
            if tile_name in info['file']:
                tile_id = tid
                tile_info = info
                break

        if tile_id is None:
            print(f"  WARNING: Could not find tile in index")
            continue

        # Basic statistics
        valid_mask = ~np.isnan(daylight)
        if not np.any(valid_mask):
            print(f"  ERROR: No valid data!")
            continue

        valid_data = daylight[valid_mask]

        stats = {
            'file': result_file.name,
            'tile_id': tile_id,
            'shape': daylight.shape,
            'valid_pixels': np.sum(valid_mask),
            'valid_percent': 100 * np.sum(valid_mask) / daylight.size,
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'median': float(np.median(valid_data)),
            'std': float(np.std(valid_data))
        }

        all_stats.append(stats)

        print(f"  Shape: {stats['shape']}")
        print(f"  Valid pixels: {stats['valid_pixels']:,} ({stats['valid_percent']:.1f}%)")
        print(f"  Daylight hours:")
        print(f"    Min:    {stats['min']:.2f}h")
        print(f"    Max:    {stats['max']:.2f}h")
        print(f"    Mean:   {stats['mean']:.2f}h")
        print(f"    Median: {stats['median']:.2f}h")
        print(f"    Std:    {stats['std']:.2f}h")

        # Data validation
        if stats['min'] < 0 or stats['max'] > 16:
            print(f"  ⚠️  WARNING: Values out of expected range [0, 16]")

        if stats['std'] < 0.1:
            print(f"  ⚠️  WARNING: Very low variance - check if data is correct")

        # Find highest daylight points in this tile
        if stats['max'] > 14.0:  # Likely peaks/ridges
            max_indices = np.where(daylight == stats['max'])

            for j in range(min(3, len(max_indices[0]))):  # Top 3 points
                row = max_indices[0][j]
                col = max_indices[1][j]

                # Get coordinates
                x_lambert, y_lambert = get_tile_coordinates(tile_info, row, col)
                lat, lon = lambert93_to_latlon(x_lambert, y_lambert)

                highest_points.append({
                    'tile': result_file.name,
                    'daylight_hours': float(daylight[row, col]),
                    'pixel_position': (int(row), int(col)),
                    'lambert93': (float(x_lambert), float(y_lambert)),
                    'latlon': (float(lat), float(lon)),
                    'google_maps': f"https://www.google.com/maps?q={lat},{lon}"
                })

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    all_mins = [s['min'] for s in all_stats]
    all_maxs = [s['max'] for s in all_stats]
    all_means = [s['mean'] for s in all_stats]

    print(f"\nAcross {len(all_stats)} tiles:")
    print(f"  Minimum daylight: {min(all_mins):.2f}h")
    print(f"  Maximum daylight: {max(all_maxs):.2f}h")
    print(f"  Average mean:     {np.mean(all_means):.2f}h")
    print(f"  Valid data:       {np.mean([s['valid_percent'] for s in all_stats]):.1f}%")

    # Highest daylight points
    if highest_points:
        print("\n" + "="*70)
        print("HIGHEST DAYLIGHT POINTS (Likely Mountain Peaks/Ridges)")
        print("="*70)

        # Sort by daylight hours
        highest_points.sort(key=lambda x: x['daylight_hours'], reverse=True)

        print(f"\nTop {min(10, len(highest_points))} points with most daylight:")
        print()

        for i, point in enumerate(highest_points[:10], 1):
            lat, lon = point['latlon']
            hours = point['daylight_hours']

            print(f"{i}. {hours:.2f}h of daylight")
            print(f"   Tile: {point['tile']}")
            print(f"   Coordinates: {lat:.6f}°N, {lon:.6f}°E")
            print(f"   Google Maps: {point['google_maps']}")
            print()

    # Edge continuity check
    print("="*70)
    print("EDGE CONTINUITY CHECK")
    print("="*70)

    print("\nChecking for discontinuities between neighboring tiles...")

    edge_issues = []

    for i, result_file in enumerate(result_files[:min(5, len(result_files))]):
        # Find tile in index
        tile_name = result_file.stem.replace('_daylight', '')
        tile_id = None
        for tid, info in spatial_index['tiles'].items():
            if tile_name in info['file']:
                tile_id = tid
                break

        if tile_id is None:
            continue

        # Load this tile's results
        daylight_a = np.load(result_file)

        # Check East neighbor
        from build_spatial_index import find_neighbors
        neighbors = find_neighbors(spatial_index, tile_id)

        if neighbors.get('E') is not None:
            neighbor_id = neighbors['E']
            neighbor_file = None

            for rf in result_files:
                if spatial_index['files'][neighbor_id].split('/')[-1].replace('.asc', '') in rf.stem:
                    neighbor_file = rf
                    break

            if neighbor_file and neighbor_file.exists():
                daylight_b = np.load(neighbor_file)

                # Compare edges
                edge_a = daylight_a[:, -1]  # Right edge
                edge_b = daylight_b[:, 0]   # Left edge

                # Calculate difference (ignoring NaN)
                mask = ~(np.isnan(edge_a) | np.isnan(edge_b))
                if np.any(mask):
                    diff = np.abs(edge_a[mask] - edge_b[mask])
                    mean_diff = np.mean(diff)
                    max_diff = np.max(diff)

                    print(f"\nTile {result_file.stem} → East neighbor:")
                    print(f"  Mean edge difference: {mean_diff:.3f}h")
                    print(f"  Max edge difference:  {max_diff:.3f}h")

                    if mean_diff > 1.0:
                        print(f"  ⚠️  WARNING: Large discontinuity detected!")
                        edge_issues.append((result_file.name, 'E', mean_diff))
                    elif mean_diff < 0.3:
                        print(f"  ✓ Edges are continuous")

    if not edge_issues:
        print("\n✓ No major edge discontinuities detected")
    else:
        print(f"\n⚠️  Found {len(edge_issues)} tiles with edge issues")

    # Save detailed report
    report_file = Path(results_dir) / "verification_report.json"
    report = {
        'num_tiles': len(all_stats),
        'overall_stats': {
            'min_daylight': float(min(all_mins)),
            'max_daylight': float(max(all_maxs)),
            'avg_mean_daylight': float(np.mean(all_means))
        },
        'tiles': all_stats,
        'highest_points': highest_points[:10],
        'edge_issues': [
            {'tile': tile, 'direction': dir, 'difference': float(diff)}
            for tile, dir, diff in edge_issues
        ]
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print(f"Detailed report saved to: {report_file}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify daylight computation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results
  python3 verify_results.py

  # Analyze first 10 tiles
  python3 verify_results.py --num-tiles 10

  # Custom directories
  python3 verify_results.py --results-dir my_results/ --index asc_spatial_index.pkl
        """
    )

    parser.add_argument(
        "--results-dir",
        default="daylight_results_tiles",
        help="Directory with result files (default: daylight_results_tiles)"
    )

    parser.add_argument(
        "--index",
        default="asc_spatial_index.pkl",
        help="Spatial index file (default: asc_spatial_index.pkl)"
    )

    parser.add_argument(
        "--num-tiles",
        type=int,
        help="Number of tiles to analyze (default: all)"
    )

    args = parser.parse_args()

    analyze_tile_results(
        results_dir=args.results_dir,
        index_file=args.index,
        num_tiles=args.num_tiles
    )
