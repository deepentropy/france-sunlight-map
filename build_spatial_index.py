#!/usr/bin/env python3
"""
Build spatial index of ASC files for neighbor-aware processing.

Scans all ASC files, extracts their spatial coordinates, and creates
an index for fast neighbor lookup during daylight computation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_asc_header(asc_path: Path) -> Optional[Dict]:
    """
    Read ASC file header with spatial metadata.

    Returns:
        Dictionary with ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value
    """
    header = {}
    try:
        with open(asc_path, 'r') as f:
            for i in range(6):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    key = line[0].lower()
                    try:
                        if key in ['ncols', 'nrows']:
                            header[key] = int(line[1])
                        else:
                            header[key] = float(line[1])
                    except ValueError:
                        continue

        # Validate required fields
        required = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize']
        if all(k in header for k in required):
            return header
        else:
            logger.warning(f"Missing required fields in {asc_path}")
            return None

    except Exception as e:
        logger.error(f"Error reading {asc_path}: {e}")
        return None


def get_tile_bounds(header: Dict) -> Tuple[float, float, float, float]:
    """
    Calculate tile bounding box from header.

    Returns:
        (xmin, ymin, xmax, ymax) in coordinate system units
    """
    xmin = header['xllcorner']
    ymin = header['yllcorner']
    xmax = xmin + header['ncols'] * header['cellsize']
    ymax = ymin + header['nrows'] * header['cellsize']
    return (xmin, ymin, xmax, ymax)


def get_tile_center(header: Dict) -> Tuple[float, float]:
    """Get tile center coordinates."""
    xmin, ymin, xmax, ymax = get_tile_bounds(header)
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


def build_spatial_index(
    root_dir: str,
    output_file: str = "asc_spatial_index.pkl",
    pattern: str = "**/*.asc"
) -> Dict:
    """
    Build spatial index of all ASC files.

    Args:
        root_dir: Root directory to scan (e.g., "RGEALTI/")
        output_file: Output pickle file for index
        pattern: Glob pattern for ASC files

    Returns:
        Spatial index dictionary
    """
    logger.info(f"Scanning ASC files in {root_dir}")

    root = Path(root_dir)
    asc_files = list(root.glob(pattern))

    if not asc_files:
        raise ValueError(f"No ASC files found in {root_dir}")

    logger.info(f"Found {len(asc_files)} ASC files")

    # Build index
    spatial_index = {
        'tiles': {},       # tile_id → metadata
        'grid': {},        # (grid_x, grid_y) → tile_id
        'bounds': {},      # tile_id → (xmin, ymin, xmax, ymax)
        'files': {},       # tile_id → file_path
        'cellsize': None,  # Common cell size
        'grid_size': None  # Tile size in grid units
    }

    tile_id = 0
    cellsizes = set()
    grid_sizes = set()

    logger.info("Reading headers and building index...")
    for asc_path in tqdm(asc_files, desc="Indexing"):
        header = read_asc_header(asc_path)

        if header is None:
            continue

        # Store tile info
        bounds = get_tile_bounds(header)
        center = get_tile_center(header)

        # Calculate grid position (normalized by tile size)
        cellsize = header['cellsize']
        tile_width = header['ncols'] * cellsize
        tile_height = header['nrows'] * cellsize

        # Grid coordinates (tiles are on a regular grid)
        grid_x = round(header['xllcorner'] / tile_width)
        grid_y = round(header['yllcorner'] / tile_height)

        spatial_index['tiles'][tile_id] = {
            'file': str(asc_path),
            'header': header,
            'bounds': bounds,
            'center': center,
            'grid_pos': (grid_x, grid_y)
        }

        spatial_index['grid'][(grid_x, grid_y)] = tile_id
        spatial_index['bounds'][tile_id] = bounds
        spatial_index['files'][tile_id] = str(asc_path)

        cellsizes.add(cellsize)
        grid_sizes.add((tile_width, tile_height))

        tile_id += 1

    # Validate consistent grid
    if len(cellsizes) > 1:
        logger.warning(f"Multiple cell sizes found: {cellsizes}")
    if len(grid_sizes) > 1:
        logger.warning(f"Multiple tile sizes found: {grid_sizes}")

    spatial_index['cellsize'] = list(cellsizes)[0] if cellsizes else None
    spatial_index['grid_size'] = list(grid_sizes)[0] if grid_sizes else None

    logger.info(f"Indexed {tile_id} tiles")
    logger.info(f"Cell size: {spatial_index['cellsize']} m")
    logger.info(f"Tile size: {spatial_index['grid_size']}")

    # Save index
    logger.info(f"Saving index to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(spatial_index, f)

    # Also save human-readable JSON (without numpy arrays)
    json_file = output_file.replace('.pkl', '.json')
    json_index = {
        'num_tiles': tile_id,
        'cellsize': spatial_index['cellsize'],
        'grid_size': spatial_index['grid_size'],
        'files': {
            tile_id: info['file']
            for tile_id, info in spatial_index['tiles'].items()
        }
    }
    with open(json_file, 'w') as f:
        json.dump(json_index, f, indent=2)

    logger.info(f"Index saved: {output_file} (pickle) and {json_file} (JSON)")

    return spatial_index


def load_spatial_index(index_file: str = "asc_spatial_index.pkl") -> Dict:
    """Load spatial index from file."""
    logger.info(f"Loading spatial index from {index_file}")
    with open(index_file, 'rb') as f:
        return pickle.load(f)


def find_neighbors(
    spatial_index: Dict,
    tile_id: int,
    search_distance: int = 1
) -> Dict[str, Optional[int]]:
    """
    Find neighboring tiles.

    Args:
        spatial_index: Spatial index dictionary
        tile_id: ID of center tile
        search_distance: How many grid cells away to search (default: 1 for 8 neighbors)

    Returns:
        Dictionary of neighbor directions → tile_id
        Directions: 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'
    """
    tile_info = spatial_index['tiles'][tile_id]
    gx, gy = tile_info['grid_pos']

    neighbors = {}

    # 8 cardinal and diagonal directions
    directions = {
        'N':  (0, 1),
        'S':  (0, -1),
        'E':  (1, 0),
        'W':  (-1, 0),
        'NE': (1, 1),
        'NW': (-1, 1),
        'SE': (1, -1),
        'SW': (-1, -1)
    }

    for direction, (dx, dy) in directions.items():
        neighbor_pos = (gx + dx, gy + dy)
        neighbor_id = spatial_index['grid'].get(neighbor_pos)
        neighbors[direction] = neighbor_id

    return neighbors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build spatial index of ASC files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all ASC files under RGEALTI/
  python3 build_spatial_index.py RGEALTI/

  # Custom output file
  python3 build_spatial_index.py RGEALTI/ --output my_index.pkl

  # Custom pattern
  python3 build_spatial_index.py RGEALTI/ --pattern "D001*/**/*.asc"
        """
    )

    parser.add_argument(
        "root_dir",
        help="Root directory containing ASC files"
    )

    parser.add_argument(
        "--output",
        default="asc_spatial_index.pkl",
        help="Output pickle file (default: asc_spatial_index.pkl)"
    )

    parser.add_argument(
        "--pattern",
        default="**/*.asc",
        help="Glob pattern for ASC files (default: **/*.asc)"
    )

    args = parser.parse_args()

    # Build index
    index = build_spatial_index(
        args.root_dir,
        args.output,
        args.pattern
    )

    # Show summary
    print(f"\n{'='*60}")
    print(f"Spatial Index Summary")
    print(f"{'='*60}")
    print(f"Total tiles: {len(index['tiles'])}")
    print(f"Cell size: {index['cellsize']} m")
    print(f"Tile dimensions: {index['grid_size']}")
    print(f"\nIndex files created:")
    print(f"  - {args.output} (pickle - for processing)")
    print(f"  - {args.output.replace('.pkl', '.json')} (JSON - for inspection)")
    print(f"{'='*60}")
