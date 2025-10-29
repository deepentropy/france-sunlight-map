#!/usr/bin/env python3
"""
Process ASC files with neighbor context for accurate shadow calculations.

Uses spatial index to find neighbors and process tiles with proper edge handling.
"""

import numpy as np
import cupy as cp
from pathlib import Path
from datetime import datetime
from typing import Tuple
import pickle
import logging
import time
from tqdm import tqdm
from build_spatial_index import load_spatial_index, find_neighbors
from compute import CuPySolarCalculator, SystemMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_asc_data(asc_path: str, dtype=np.float32) -> np.ndarray:
    """Load ASC data, skipping header."""
    data = np.loadtxt(asc_path, skiprows=6, dtype=dtype)
    # Replace NODATA with NaN
    data[data == -99999] = np.nan
    return data


def create_extended_tile(
    center_tile: np.ndarray,
    neighbors: dict,
    spatial_index: dict,
    tile_id: int,
    overlap: int = 100
) -> Tuple[np.ndarray, dict]:
    """
    Create extended tile with neighbor context for shadow calculations.

    Args:
        center_tile: Central tile array
        neighbors: Dictionary of neighbor tile_ids
        spatial_index: Spatial index
        tile_id: Current tile ID
        overlap: Number of pixels to include from each neighbor

    Returns:
        (extended_array, offsets_dict)
        extended_array: Larger array with tile + neighbor edges
        offsets_dict: Where center tile starts in extended array
    """
    h, w = center_tile.shape

    # Calculate extended dimensions
    # We need overlap pixels from each direction
    ext_h = h + 2 * overlap  # overlap on top and bottom
    ext_w = w + 2 * overlap  # overlap on left and right

    # Create extended array filled with NaN
    extended = np.full((ext_h, ext_w), np.nan, dtype=center_tile.dtype)

    # Place center tile
    extended[overlap:overlap+h, overlap:overlap+w] = center_tile

    # Load and place neighbor edges
    # North neighbor (top edge)
    if neighbors.get('N') is not None:
        try:
            n_path = spatial_index['files'][neighbors['N']]
            n_data = load_asc_data(n_path)
            # Take bottom 'overlap' rows from north neighbor
            extended[0:overlap, overlap:overlap+w] = n_data[-overlap:, :]
        except Exception as e:
            logger.debug(f"Could not load N neighbor: {e}")

    # South neighbor (bottom edge)
    if neighbors.get('S') is not None:
        try:
            s_path = spatial_index['files'][neighbors['S']]
            s_data = load_asc_data(s_path)
            # Take top 'overlap' rows from south neighbor
            extended[overlap+h:ext_h, overlap:overlap+w] = s_data[:overlap, :]
        except Exception as e:
            logger.debug(f"Could not load S neighbor: {e}")

    # East neighbor (right edge)
    if neighbors.get('E') is not None:
        try:
            e_path = spatial_index['files'][neighbors['E']]
            e_data = load_asc_data(e_path)
            # Take left 'overlap' columns from east neighbor
            extended[overlap:overlap+h, overlap+w:ext_w] = e_data[:, :overlap]
        except Exception as e:
            logger.debug(f"Could not load E neighbor: {e}")

    # West neighbor (left edge)
    if neighbors.get('W') is not None:
        try:
            w_path = spatial_index['files'][neighbors['W']]
            w_data = load_asc_data(w_path)
            # Take right 'overlap' columns from west neighbor
            extended[overlap:overlap+h, 0:overlap] = w_data[:, -overlap:]
        except Exception as e:
            logger.debug(f"Could not load W neighbor: {e}")

    # Corners (NE, NW, SE, SW)
    # Northeast corner
    if neighbors.get('NE') is not None:
        try:
            ne_path = spatial_index['files'][neighbors['NE']]
            ne_data = load_asc_data(ne_path)
            extended[0:overlap, overlap+w:ext_w] = ne_data[-overlap:, :overlap]
        except Exception as e:
            logger.debug(f"Could not load NE neighbor: {e}")

    # Northwest corner
    if neighbors.get('NW') is not None:
        try:
            nw_path = spatial_index['files'][neighbors['NW']]
            nw_data = load_asc_data(nw_path)
            extended[0:overlap, 0:overlap] = nw_data[-overlap:, -overlap:]
        except Exception as e:
            logger.debug(f"Could not load NW neighbor: {e}")

    # Southeast corner
    if neighbors.get('SE') is not None:
        try:
            se_path = spatial_index['files'][neighbors['SE']]
            se_data = load_asc_data(se_path)
            extended[overlap+h:ext_h, overlap+w:ext_w] = se_data[:overlap, :overlap]
        except Exception as e:
            logger.debug(f"Could not load SE neighbor: {e}")

    # Southwest corner
    if neighbors.get('SW') is not None:
        try:
            sw_path = spatial_index['files'][neighbors['SW']]
            sw_data = load_asc_data(sw_path)
            extended[overlap+h:ext_h, 0:overlap] = sw_data[:overlap, -overlap:]
        except Exception as e:
            logger.debug(f"Could not load SW neighbor: {e}")

    offsets = {
        'row_start': overlap,
        'row_end': overlap + h,
        'col_start': overlap,
        'col_end': overlap + w
    }

    return extended, offsets


def process_all_tiles_with_neighbors(
    spatial_index_file: str,
    output_dir: str = "daylight_results_tiles",
    date: datetime = None,
    pixel_size: float = 5.0,
    lat_center: float = 46.0,
    lon_center: float = 2.0,
    overlap: int = 100,
    max_tiles: int = None
):
    """
    Process all tiles with neighbor context.

    Args:
        spatial_index_file: Path to spatial index pickle file
        output_dir: Output directory for results
        date: Date for sun calculations
        pixel_size: Cell size in meters
        lat_center: Latitude of center
        lon_center: Longitude of center
        overlap: Pixels to load from neighbors AND maximum shadow tracing distance
                 At 5m resolution:
                 - 100px = 500m (default, good for flat terrain)
                 - 500px = 2.5km (hilly terrain)
                 - 1000px = 5km (mountainous)
                 - 2000px = 10km (high mountains like Alps, Corsica)
        max_tiles: Maximum tiles to process (for testing)
    """
    from pathlib import Path
    import os

    date = date or datetime(2024, 6, 21)

    # Load spatial index
    logger.info("Loading spatial index...")
    spatial_index = load_spatial_index(spatial_index_file)

    num_tiles = len(spatial_index['tiles'])
    logger.info(f"Found {num_tiles} tiles in index")

    if max_tiles:
        num_tiles = min(num_tiles, max_tiles)
        logger.info(f"Processing first {num_tiles} tiles (max_tiles limit)")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize GPU calculator
    gpu_calc = CuPySolarCalculator()

    # Pre-calculate sun positions
    logger.info(f"Center coordinates: {lat_center:.4f}°N, {lon_center:.4f}°E")
    times = []
    start_hour = datetime.combine(date, datetime.min.time()).replace(hour=5)
    for hour in range(5, 21):
        for minute in [0, 30]:
            times.append(start_hour.replace(hour=hour, minute=minute))

    elevations, azimuths = gpu_calc.solar_position_vectorized(
        lat_center, lon_center, times, date
    )
    sun_positions = list(zip(elevations, azimuths))

    logger.info(f"Processing {num_tiles} tiles with {overlap}px neighbor overlap")
    logger.info(f"Shadow search distance: {overlap * pixel_size:.0f}m")

    # Process each tile
    processed = 0
    start_time = time.time()

    for tile_id in tqdm(range(num_tiles), desc="Processing tiles"):
        try:
            tile_info = spatial_index['tiles'][tile_id]
            asc_path = tile_info['file']

            # Load center tile
            center_tile = load_asc_data(asc_path)

            # Find neighbors
            neighbors = find_neighbors(spatial_index, tile_id)

            # Create extended tile with neighbor context
            extended_tile, offsets = create_extended_tile(
                center_tile, neighbors, spatial_index, tile_id, overlap
            )

            # Process extended tile on GPU
            # Use overlap as max_shadow_distance to utilize full loaded context
            daylight_extended = gpu_calc.compute_shadows_batch_gpu(
                [extended_tile], sun_positions, pixel_size, max_shadow_distance=overlap
            )[0]

            # Extract center region (original tile)
            daylight_center = daylight_extended[
                offsets['row_start']:offsets['row_end'],
                offsets['col_start']:offsets['col_end']
            ]

            # Save result
            output_name = Path(asc_path).stem + "_daylight.npy"
            output_path = os.path.join(output_dir, output_name)
            np.save(output_path, daylight_center)

            processed += 1

            # Clear memory
            del center_tile, extended_tile, daylight_extended, daylight_center
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            logger.error(f"Error processing tile {tile_id} ({asc_path}): {e}")
            continue

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Processed: {processed}/{num_tiles} tiles")
    logger.info(f"Time: {elapsed/60:.2f} minutes")
    logger.info(f"Speed: {processed/elapsed:.2f} tiles/second")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ASC tiles with neighbor context for accurate daylight computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tiles (default: 500m shadow distance)
  python3 process_asc_with_neighbors.py asc_spatial_index.pkl

  # Mountainous terrain like Corsica (10km shadow distance)
  python3 process_asc_with_neighbors.py asc_spatial_index.pkl --overlap 2000

  # Test with first 10 tiles
  python3 process_asc_with_neighbors.py asc_spatial_index.pkl --max-tiles 10 --overlap 2000

  # Alps/Pyrenees (5km shadow distance)
  python3 process_asc_with_neighbors.py asc_spatial_index.pkl --overlap 1000

Shadow distance guide (at 5m resolution):
  --overlap 100  =  500m   (flat terrain)
  --overlap 500  = 2.5km   (hilly)
  --overlap 1000 =   5km   (mountains)
  --overlap 2000 =  10km   (high mountains)
        """
    )

    parser.add_argument(
        "index_file",
        help="Spatial index pickle file (from build_spatial_index.py)"
    )

    parser.add_argument(
        "--output-dir",
        default="daylight_results_tiles",
        help="Output directory (default: daylight_results_tiles)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Shadow search distance in pixels. Also loads this much from neighbors. "
             "(default: 100=500m, use 2000=10km for Corsica/high mountains)"
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=5.0,
        help="Cell size in meters (default: 5.0)"
    )

    parser.add_argument(
        "--lat-center",
        type=float,
        default=46.0,
        help="Latitude of center point (default: 46.0)"
    )

    parser.add_argument(
        "--lon-center",
        type=float,
        default=2.0,
        help="Longitude of center point (default: 2.0)"
    )

    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Maximum tiles to process (for testing)"
    )

    args = parser.parse_args()

    process_all_tiles_with_neighbors(
        spatial_index_file=args.index_file,
        output_dir=args.output_dir,
        date=datetime(2024, 6, 21),
        pixel_size=args.pixel_size,
        lat_center=args.lat_center,
        lon_center=args.lon_center,
        overlap=args.overlap,
        max_tiles=args.max_tiles
    )
