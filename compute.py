import numpy as np
import cupy as cp
import os
import glob
from datetime import datetime, timedelta
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_calculation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources"""

    @staticmethod
    def log_gpu_status():
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            total_memory = device.mem_info[1] / 1e9
            free_memory = device.mem_info[0] / 1e9
            used_memory = total_memory - free_memory

            logger.info(
                f"GPU Memory: {used_memory:.1f}/{total_memory:.1f}GB ({(used_memory / total_memory) * 100:.1f}%)")
        except Exception as e:
            logger.debug(f"Could not get GPU status: {e}")

    @staticmethod
    def log_system_status():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        logger.info(f"CPU: {cpu_percent}% | RAM: {mem.percent}% ({mem.used / 1e9:.1f}/{mem.total / 1e9:.1f}GB)")


class OptimizedASCReader:
    """Optimized ASC file reader"""

    @staticmethod
    def read_header_fast(asc_path):
        """Read only header without loading data"""
        header = {}
        try:
            with open(asc_path, 'r', buffering=8192) as f:
                for i in range(6):
                    line = f.readline().strip().split()
                    if len(line) >= 2:
                        key = line[0].lower()
                        value = line[1]
                        if key in ['ncols', 'nrows']:
                            header[key] = int(value)
                        elif key in ['xllcorner', 'yllcorner', 'cellsize']:
                            header[key] = float(value)
                        elif key == 'nodata_value':
                            header[key] = float(value)
            return header
        except Exception as e:
            logger.error(f"Error reading header: {e}")
            return None

    @staticmethod
    def read_data_fast(asc_path, dtype=np.float32):
        """Optimized data reading"""
        start_time = time.time()

        header = OptimizedASCReader.read_header_fast(asc_path)
        if not header:
            return None, None

        try:
            # Read data efficiently
            data = np.loadtxt(asc_path, skiprows=6, dtype=dtype)

            # Replace nodata values
            if 'nodata_value' in header:
                data[data == header['nodata_value']] = np.nan

            elapsed = time.time() - start_time
            logger.debug(f"Loaded in {elapsed:.2f}s - Shape: {data.shape}")

            return data, header

        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return None, None


class CuPySolarCalculator:
    """GPU-accelerated solar calculator using only CuPy"""

    def __init__(self):
        self.gpu_available = self.check_gpu()

    def check_gpu(self):
        """Check GPU availability"""
        try:
            # Test CuPy
            test_array = cp.array([1, 2, 3])
            result = cp.sum(test_array)

            device = cp.cuda.Device()
            logger.info(f"GPU available: RTX 4090 - Memory: {device.mem_info[1] / 1e9:.1f}GB")
            return True

        except Exception as e:
            logger.warning(f"GPU not available: {e}")
            return False

    def solar_position_vectorized(self, lat, lon, times, date):
        """Vectorized solar position calculation"""
        n = date.timetuple().tm_yday
        declination = 23.45 * np.sin(np.radians((360 * (284 + n)) / 365))

        hours = np.array([t.hour + t.minute / 60 for t in times])
        hour_angles = 15 * (hours - 12)

        lat_rad = np.radians(lat)
        dec_rad = np.radians(declination)
        hour_rad = np.radians(hour_angles)

        elevations = np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        )

        azimuths = np.arctan2(
            -np.sin(hour_rad),
            np.tan(lat_rad) * np.cos(hour_rad) - np.tan(dec_rad)
        )

        return np.degrees(elevations), np.degrees(azimuths)

    def compute_shadows_gpu_cupy(self, elevation_np, sun_positions, pixel_size=5.0):
        """Compute shadows using CuPy (pure GPU arrays) - GPU ONLY MODE"""
        if not self.gpu_available:
            raise RuntimeError("GPU is required for computation. CPU fallback has been disabled for full accuracy.")

        try:
            height, width = elevation_np.shape

            # Transfer elevation to GPU once
            elevation_gpu = cp.asarray(elevation_np, dtype=cp.float32)
            total_daylight_gpu = cp.zeros((height, width), dtype=cp.float32)

            # Pre-compute indices for efficiency
            i_indices_gpu = cp.arange(height).reshape(-1, 1)
            j_indices_gpu = cp.arange(width).reshape(1, -1)

            processed = 0

            for sun_elev, sun_azim in tqdm(sun_positions, desc="Computing shadows (GPU)", leave=False):
                if sun_elev <= 0:
                    continue

                processed += 1

                # Convert angles to radians
                sun_elev_rad = cp.radians(sun_elev)
                sun_azim_rad = cp.radians(sun_azim)

                # Ray direction
                dx = -cp.sin(sun_azim_rad)
                dy = cp.cos(sun_azim_rad)
                tan_elev = cp.tan(sun_elev_rad)

                # Initialize shadow map
                shadows_gpu = cp.ones((height, width), dtype=cp.float32)

                # Mark NaN areas as shadowed
                shadows_gpu[cp.isnan(elevation_gpu)] = 0

                # Compute shadows using vectorized operations
                # Increased distance for better accuracy
                max_dist = min(100, min(height, width) // 2)

                for d in range(1, max_dist):  # Check EVERY distance (no skipping)
                    # Source positions
                    source_i = (i_indices_gpu + int(d * dy)).astype(cp.int32)
                    source_j = (j_indices_gpu + int(d * dx)).astype(cp.int32)

                    # Valid mask
                    valid_mask = (source_i >= 0) & (source_i < height) & \
                                 (source_j >= 0) & (source_j < width)

                    # Calculate required elevation
                    dist_m = d * pixel_size
                    required_elev_diff = dist_m * tan_elev

                    # Get terrain elevations at source positions
                    # Use advanced indexing with bounds checking
                    source_i_clipped = cp.clip(source_i, 0, height - 1)
                    source_j_clipped = cp.clip(source_j, 0, width - 1)

                    source_elevations = elevation_gpu[source_i_clipped, source_j_clipped]

                    # Check if terrain blocks sun
                    blocked = valid_mask & (source_elevations > (elevation_gpu + required_elev_diff))
                    shadows_gpu[blocked] = 0

                # Accumulate daylight
                total_daylight_gpu += shadows_gpu * 0.5  # 30 min steps

            # Transfer back to CPU
            total_daylight = cp.asnumpy(total_daylight_gpu)

            # Clean up GPU memory
            del elevation_gpu
            del total_daylight_gpu
            cp.get_default_memory_pool().free_all_blocks()

            logger.debug(f"Processed {processed} sun positions on GPU")
            return total_daylight

        except Exception as e:
            logger.error(f"GPU computation failed: {e}")
            raise RuntimeError(f"GPU computation failed and CPU fallback is disabled. Error: {e}")


class BatchProcessor:
    """Batch processing manager"""

    def __init__(self, asc_directory):
        self.asc_directory = asc_directory
        self.gpu_calc = CuPySolarCalculator()

    def process_all(self, date=None, max_tiles=None, batch_size=5):
        """Process all tiles in batches"""
        date = date or datetime(2024, 6, 21)

        # Find ASC files
        asc_files = glob.glob(os.path.join(self.asc_directory, "*.asc"))

        if not asc_files:
            logger.error(f"No ASC files found")
            return []

        if max_tiles:
            asc_files = asc_files[:max_tiles]

        total = len(asc_files)
        logger.info(f"Processing {total} ASC files")

        # Get center coordinates
        lat_center, lon_center = self.get_center_coords(asc_files[0])
        logger.info(f"Center: {lat_center:.4f}°N, {lon_center:.4f}°E")

        # Pre-calculate sun positions
        times = []
        start_hour = datetime.combine(date, datetime.min.time()).replace(hour=5)
        for hour in range(5, 21):
            for minute in [0, 30]:
                times.append(start_hour.replace(hour=hour, minute=minute))

        elevations, azimuths = self.gpu_calc.solar_position_vectorized(
            lat_center, lon_center, times, date
        )
        sun_positions = list(zip(elevations, azimuths))

        # Process in batches
        results = []

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_files = asc_files[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // batch_size + 1}/{(total - 1) // batch_size + 1}")

            for idx, asc_path in enumerate(batch_files, batch_start + 1):
                try:
                    logger.info(f"[{idx}/{total}] Processing {os.path.basename(asc_path)}")
                    start_time = time.time()

                    # Read data
                    elevation, header = OptimizedASCReader.read_data_fast(asc_path)
                    if elevation is None:
                        continue

                    # Compute shadows using CuPy
                    daylight_hours = self.gpu_calc.compute_shadows_gpu_cupy(
                        elevation, sun_positions, header['cellsize']
                    )

                    elapsed = time.time() - start_time
                    logger.info(f"[{idx}/{total}] Completed in {elapsed:.2f}s")

                    # Store result
                    results.append({
                        'filename': os.path.basename(asc_path),
                        'daylight': daylight_hours,
                        'header': header
                    })

                except Exception as e:
                    logger.error(f"Error processing {asc_path}: {e}")

            # Clear GPU memory after each batch
            cp.get_default_memory_pool().free_all_blocks()

            # Log status
            SystemMonitor.log_system_status()
            SystemMonitor.log_gpu_status()

        logger.info(f"Processed {len(results)}/{total} tiles successfully")
        return results

    def get_center_coords(self, sample_file):
        """Get center coordinates"""
        try:
            header = OptimizedASCReader.read_header_fast(sample_file)
            if header:
                transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
                x = header['xllcorner'] + (header['ncols'] * header['cellsize']) / 2
                y = header['yllcorner'] + (header['nrows'] * header['cellsize']) / 2
                lon, lat = transformer.transform(x, y)
                return lat, lon
        except Exception as e:
            logger.warning(f"Using default coords: {e}")
        return 46.0, 2.0


def save_results(results, output_dir="daylight_results"):
    """Save all results"""
    os.makedirs(output_dir, exist_ok=True)

    for result in tqdm(results, desc="Saving results"):
        if result:
            filename = result['filename'].replace('.asc', '_daylight.npy')
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, result['daylight'])

    logger.info(f"Results saved to {output_dir}")


def create_visualization(results, output_file="daylight_map.png"):
    """Create visualization"""
    if not results:
        return

    # Visualize first result
    first_result = results[0]
    daylight = first_result['daylight']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Daylight map
    im1 = axes[0].imshow(daylight, cmap='YlOrRd', aspect='equal', vmin=0, vmax=16)
    axes[0].set_title(f"Daylight Hours - {first_result['filename']}")
    plt.colorbar(im1, ax=axes[0], label='Hours')

    # Histogram
    valid_data = daylight[daylight > 0].flatten()
    if len(valid_data) > 0:
        axes[1].hist(valid_data, bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Hours of daylight')
        axes[1].set_ylabel('Number of pixels')
        axes[1].set_title('Distribution of daylight hours')
        axes[1].axvline(valid_data.mean(), color='red', linestyle='--', label=f'Mean: {valid_data.mean():.1f}h')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_file}")

    # Statistics
    if len(valid_data) > 0:
        logger.info(f"Statistics for first tile:")
        logger.info(f"  Min: {valid_data.min():.1f}h")
        logger.info(f"  Max: {valid_data.max():.1f}h")
        logger.info(f"  Mean: {valid_data.mean():.1f}h")
        logger.info(f"  Median: {np.median(valid_data):.1f}h")


def main():
    """Main execution"""

    # Configuration
    ASC_DIR = "RGEALTI/1_DONNEES_LIVRAISON_2021-10-00009/RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D074"

    if not os.path.exists(ASC_DIR):
        ASC_DIR = input("Enter ASC directory path: ").strip()
        if not os.path.exists(ASC_DIR):
            logger.error(f"Directory not found: {ASC_DIR}")
            return

    # Parameters
    MAX_TILES = None  # Process ALL tiles (no limit)
    BATCH_SIZE = 5  # Process 5 files at a time to manage memory

    logger.info("=" * 60)
    logger.info("Solar Daylight Calculation with CuPy GPU Acceleration")
    logger.info(f"Directory: {ASC_DIR}")
    logger.info(f"Processing up to {MAX_TILES or 'all'} tiles")
    logger.info("=" * 60)

    # Log system status
    SystemMonitor.log_system_status()

    # Create processor
    processor = BatchProcessor(ASC_DIR)

    # Process tiles
    start_time = time.time()
    results = processor.process_all(
        date=datetime(2024, 6, 21),
        max_tiles=MAX_TILES,
        batch_size=BATCH_SIZE
    )
    elapsed = time.time() - start_time

    logger.info(f"Total processing time: {elapsed / 60:.2f} minutes")

    if results:
        # Save results
        save_results(results)

        # Create visualization
        create_visualization(results)

        logger.info("Processing complete!")

        # Estimate time for all files
        if MAX_TILES and len(results) == MAX_TILES:
            time_per_file = elapsed / MAX_TILES
            estimated_total = time_per_file * 235 / 60
            logger.info(f"Estimated time for all 235 files: {estimated_total:.1f} minutes")

            response = input(f"\nProcess all 235 files? (y/n): ")
            if response.lower() == 'y':
                logger.info("Processing all files...")
                all_results = processor.process_all(
                    date=datetime(2024, 6, 21),
                    max_tiles=None,
                    batch_size=BATCH_SIZE
                )
                save_results(all_results)
                logger.info(f"Processed {len(all_results)} files total")


if __name__ == "__main__":
    main()