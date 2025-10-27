import numpy as np
import os
import glob
from pyproj import Transformer
import folium
from folium import plugins
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU for image processing")


class DaylightMapVisualizer:
    """Create interactive maps from NPY daylight data with GPU acceleration"""

    def __init__(self, results_dir="daylight_results", asc_dir=None):
        self.results_dir = results_dir
        self.asc_dir = asc_dir
        self.npy_files = sorted(glob.glob(os.path.join(results_dir, "*_daylight.npy")))
        self.transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        self.metadata = {}

        print(f"Found {len(self.npy_files)} NPY files in {results_dir}")

        if asc_dir and os.path.exists(asc_dir):
            self.load_metadata()
        else:
            print(f"Warning: ASC directory not found: {asc_dir}")

    def load_metadata(self):
        """Load metadata from ASC files with correct coordinate transformations"""
        asc_files = sorted(glob.glob(os.path.join(self.asc_dir, "*.asc")))
        print(f"Loading metadata from {len(asc_files)} ASC files...")

        for asc_file in tqdm(asc_files, desc="Loading metadata"):
            basename = os.path.basename(asc_file).replace('.asc', '')
            header = self.read_asc_header(asc_file)
            if header:
                cellsize = header['cellsize']

                # Calculate pixel CENTER coordinates in Lambert93
                # xllcorner, yllcorner represent the LOWER-LEFT corner of the grid
                # First pixel center (southwest corner):
                xmin = header['xllcorner'] + 0.5 * cellsize
                ymin = header['yllcorner'] + 0.5 * cellsize

                # Last pixel center (northeast corner):
                xmax = header['xllcorner'] + (header['ncols'] - 0.5) * cellsize
                ymax = header['yllcorner'] + (header['nrows'] - 0.5) * cellsize

                # Transform all four corners from Lambert93 to WGS84
                # Note: transformer returns (lon, lat) when always_xy=True
                lon_sw, lat_sw = self.transformer.transform(xmin, ymin)  # Southwest
                lon_se, lat_se = self.transformer.transform(xmax, ymin)  # Southeast
                lon_nw, lat_nw = self.transformer.transform(xmin, ymax)  # Northwest
                lon_ne, lat_ne = self.transformer.transform(xmax, ymax)  # Northeast

                # Find bounding box that contains all four corners
                all_lats = [lat_sw, lat_se, lat_nw, lat_ne]
                all_lons = [lon_sw, lon_se, lon_nw, lon_ne]

                south = min(all_lats)
                north = max(all_lats)
                west = min(all_lons)
                east = max(all_lons)

                self.metadata[basename] = {
                    'header': header,
                    'bounds_lambert': (xmin, ymin, xmax, ymax),
                    'bounds_wgs84': [[south, west], [north, east]],  # Folium format: [[south, west], [north, east]]
                    'center_wgs84': ((south + north) / 2, (west + east) / 2)  # (lat, lon) format
                }

        print(f"Loaded metadata for {len(self.metadata)} tiles")

    def read_asc_header(self, asc_path):
        """Read ASC header"""
        header = {}
        try:
            with open(asc_path, 'r') as f:
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
            print(f"Error reading {asc_path}: {e}")
            return None

    def create_map(self, output_html="daylight_map.html", min_hours=0, downsample=1,
                   markers_per_tile=5, max_tiles=None):
        """
        Create full-resolution interactive map

        Args:
            output_html: Output filename
            min_hours: Minimum daylight hours to display
            downsample: Downsampling factor for overlay images (1=full res, 5=web-friendly)
            markers_per_tile: Number of markers to show per tile (default 5, use 1-2 for lightweight)
            max_tiles: Maximum tiles to process (None=all, use 50 for debugging)
        """

        if not self.npy_files:
            print("No NPY files found!")
            return

        if not self.metadata:
            print("No metadata available - need ASC files for georeferencing")
            return

        # Calculate map center
        all_centers = [meta['center_wgs84'] for meta in self.metadata.values()]
        if all_centers:
            all_lats = [c[0] for c in all_centers]
            all_lons = [c[1] for c in all_centers]
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
        else:
            center_lat, center_lon = 46.0, 6.0

        print(f"Map center: {center_lat:.4f}°N, {center_lon:.4f}°E")

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            max_zoom=18,
            control_scale=True
        )

        # Add base layers
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='Street Map',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='OpenTopoMap',
            name='Topographic',
            overlay=False,
            control=True
        ).add_to(m)

        # Process tiles and create overlays
        tiles_to_process = self.npy_files[:max_tiles] if max_tiles else self.npy_files
        print(f"Creating map overlays for {len(tiles_to_process)} tiles (downsample={downsample}x)...")
        processed_count = 0
        total_pixels = 0

        for npy_file in tqdm(tiles_to_process, desc="Processing tiles"):
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                print(f"Skipping {basename} - no metadata")
                continue

            # Load daylight data
            daylight = np.load(npy_file)
            total_pixels += daylight.size
            meta = self.metadata[basename]

            # Downsample if requested
            if downsample > 1:
                daylight = daylight[::downsample, ::downsample]

            # Create overlay image
            overlay_img = self.create_overlay_image(daylight, min_hours)

            if overlay_img:
                folium.raster_layers.ImageOverlay(
                    image=overlay_img,
                    bounds=meta['bounds_wgs84'],
                    opacity=0.6,
                    name=f'Tile {basename}',
                    overlay=True,
                    control=False,
                    interactive=False,
                    cross_origin=False,
                    zindex=1
                ).add_to(m)
                processed_count += 1

        print(f"Added {processed_count} tile overlays ({total_pixels:,} pixels total)")

        # Add markers for optimal locations
        self.add_optimal_markers(m, min_hours=min_hours, markers_per_tile=markers_per_tile, max_tiles=max_tiles)

        # Add heatmap
        self.add_heatmap_layer(m, min_hours=min_hours)

        # Add legend
        self.add_legend(m)

        # Add statistics
        stats = self.calculate_statistics()
        stats_html = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; 
                    background: white; 
                    padding: 15px; 
                    border: 2px solid #ccc; 
                    border-radius: 5px;
                    z-index: 9999;
                    font-family: Arial, sans-serif;">
            <h4 style="margin-top: 0; color: #e67e22;">Daylight Hours Map</h4>
            <b>Tiles:</b> {len(self.metadata)}<br>
            <b>Data Points:</b> {total_pixels:,}<br>
            <b>Min:</b> {stats['min']:.1f}h | <b>Max:</b> {stats['max']:.1f}h<br>
            <b>Mean:</b> {stats['mean']:.1f}h<br>
            <small>Full resolution - no filtering</small>
        </div>
        """
        m.get_root().html.add_child(folium.Element(stats_html))

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # Save map
        m.save(output_html)
        print(f"Map saved to {output_html}")

        return m

    def create_overlay_image(self, daylight_data, min_hours=0):
        """Create RGBA PNG image from daylight data with GPU acceleration"""
        try:
            # Filter data
            valid_mask = (daylight_data > min_hours) & (~np.isnan(daylight_data))

            if not np.any(valid_mask):
                return None

            # Create custom colormap (blue to yellow to red)
            colors = [(0.17, 0.24, 0.31), (0.95, 0.77, 0.06), (0.91, 0.30, 0.24)]
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('daylight', colors, N=n_bins)

            # Normalize to 0-16 hours range
            norm_data = np.clip(daylight_data, 0, 16) / 16.0

            # Use GPU if available
            if GPU_AVAILABLE:
                norm_data_gpu = cp.asarray(norm_data)
                rgba_gpu = cp.zeros((norm_data.shape[0], norm_data.shape[1], 4), dtype=cp.uint8)

                for i in range(norm_data.shape[0]):
                    for j in range(norm_data.shape[1]):
                        if valid_mask[i, j]:
                            color = cmap(float(norm_data[i, j]))
                            rgba_gpu[i, j] = cp.array([
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                                180  # Semi-transparent
                            ])

                rgba = cp.asnumpy(rgba_gpu)
            else:
                # CPU processing
                rgba = np.zeros((daylight_data.shape[0], daylight_data.shape[1], 4), dtype=np.uint8)

                for i in range(daylight_data.shape[0]):
                    for j in range(daylight_data.shape[1]):
                        if valid_mask[i, j]:
                            color = cmap(norm_data[i, j])
                            rgba[i, j] = [
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                                180  # Semi-transparent
                            ]

            # Convert to PNG
            img = Image.fromarray(rgba, mode='RGBA')
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode()
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            print(f"Error creating overlay image: {e}")
            return None

    def add_legend(self, map_obj):
        """Add color legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; 
                    background: white; 
                    padding: 10px; 
                    border: 2px solid grey; 
                    z-index: 9998;
                    font-size: 14px;">
            <div style="display: flex; align-items: center;">
                <span>0h</span>
                <div style="width: 200px; height: 20px; 
                            background: linear-gradient(to right, 
                                rgba(44,62,80,0.8), 
                                rgba(241,196,15,0.8), 
                                rgba(231,76,60,0.8));
                            margin: 0 10px;"></div>
                <span>16h</span>
            </div>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def add_optimal_markers(self, map_obj, min_hours=14, markers_per_tile=5, max_tiles=None):
        """Add markers for optimal sunlight locations"""
        marker_group = folium.FeatureGroup(name='Optimal Locations (>14h)', show=True)

        marker_count = 0
        debug_count = 0
        tiles_to_process = self.npy_files[:max_tiles] if max_tiles else self.npy_files

        for npy_file in tiles_to_process:
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            daylight = np.load(npy_file)
            meta = self.metadata[basename]
            header = meta['header']

            # Find local maxima using improved algorithm
            maxima = self.find_local_maxima(daylight, threshold=min_hours, window_size=50)

            for i, j in maxima[:markers_per_tile]:
                # Convert array indices (i, j) to geographic coordinates
                #
                # Array indexing convention:
                #   - i is row (0 = top of array)
                #   - j is column (0 = left of array)
                #
                # ASC raster convention:
                #   - Data starts from NORTH (top) and goes SOUTH (bottom)
                #   - xllcorner, yllcorner are SOUTHWEST corner (lower-left)
                #   - Rows go from north to south
                #   - Columns go from west to east
                #
                # Lambert93 coordinate calculation:
                #   - X (easting) = xllcorner + (column + 0.5) * cellsize
                #   - Y (northing) = yllcorner + (nrows - row - 0.5) * cellsize
                #
                # The formula (nrows - i - 0.5) correctly converts:
                #   - Row 0 (top) → maximum Y (northernmost)
                #   - Row nrows-1 (bottom) → minimum Y (southernmost)

                x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']
                y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']

                # Transform to WGS84 (returns lon, lat)
                lon, lat = self.transformer.transform(x_lambert, y_lambert)
                hours = float(daylight[i, j])

                # Debug output for first few markers
                if debug_count < 5:
                    print(f"\n=== Marker #{debug_count + 1} ===")
                    print(f"Tile: {basename}")
                    print(f"Array indices: i={i}, j={j}")
                    print(f"Grid dimensions: nrows={header['nrows']}, ncols={header['ncols']}")
                    print(f"Lower-left corner: xllcorner={header['xllcorner']}, yllcorner={header['yllcorner']}")
                    print(f"Cellsize: {header['cellsize']}")
                    print(f"Lambert93: X={x_lambert:.1f}, Y={y_lambert:.1f}")
                    print(f"WGS84: {lat:.6f}°N, {lon:.6f}°E")
                    print(f"Daylight: {hours:.2f}h")
                    debug_count += 1

                popup_html = f"""
                <div style="width: 240px;">
                    <h4 style="color: #e67e22;">☀ Optimal Sunlight Location</h4>
                    <b>Daylight:</b> {hours:.2f}h<br>
                    <b>Array Position:</b> Row {i}, Col {j}<br>
                    <b>Lambert93:</b> {x_lambert:.1f}, {y_lambert:.1f}<br>
                    <b>WGS84:</b> {lat:.6f}°N, {lon:.6f}°E<br>
                    <b>Tile:</b> {basename}<br>
                    <hr>
                    <small>Full resolution data</small>
                </div>
                """

                icon_color = 'red' if hours >= 15 else 'orange' if hours >= 14.5 else 'yellow'

                folium.Marker(
                    location=[lat, lon],  # Folium expects [lat, lon]
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"{hours:.2f}h daylight",
                    icon=folium.Icon(color=icon_color, icon='sun', prefix='fa')
                ).add_to(marker_group)
                marker_count += 1

        marker_group.add_to(map_obj)
        print(f"Added {marker_count} optimal location markers")

    def find_local_maxima(self, data, threshold=14, window_size=50):
        """Find local maxima in daylight data with improved spacing"""
        mask = data >= threshold

        if not np.any(mask):
            return []

        from scipy.ndimage import maximum_filter

        # Use larger window to spread markers more naturally
        local_max = maximum_filter(data, size=window_size)
        maxima = (data == local_max) & mask

        coords = np.column_stack(np.where(maxima))
        values = [data[i, j] for i, j in coords]
        sorted_indices = np.argsort(values)[::-1]

        return coords[sorted_indices].tolist()

    def add_heatmap_layer(self, map_obj, min_hours=12):
        """Add heatmap layer with sample points"""
        heat_data = []

        for npy_file in self.npy_files[:50]:  # Sample from first 50 tiles
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            daylight = np.load(npy_file)
            meta = self.metadata[basename]
            header = meta['header']

            # Sample high-sunlight points
            high_sun_mask = daylight >= min_hours
            indices = np.where(high_sun_mask)

            if len(indices[0]) > 0:
                # Sample points
                n_samples = min(100, len(indices[0]))
                sample_idx = np.random.choice(len(indices[0]), n_samples, replace=False)

                for idx in sample_idx:
                    i, j = indices[0][idx], indices[1][idx]

                    # Same coordinate transformation as markers
                    x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']
                    y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']
                    lon, lat = self.transformer.transform(x_lambert, y_lambert)

                    weight = float(daylight[i, j]) / 16.0
                    heat_data.append([lat, lon, weight])

        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Daylight Density Heatmap',
                min_opacity=0.4,
                max_zoom=18,
                radius=20,
                blur=15,
                gradient={
                    0.0: '#2c3e50',
                    0.5: '#f1c40f',
                    1.0: '#e74c3c'
                },
                overlay=True,
                control=True,
                show=False
            ).add_to(map_obj)
            print(f"Added heatmap with {len(heat_data)} sample points")

    def calculate_statistics(self):
        """Calculate statistics across all NPY files"""
        all_values = []

        for npy_file in self.npy_files:
            daylight = np.load(npy_file)
            valid_data = daylight[~np.isnan(daylight) & (daylight > 0)]
            all_values.extend(valid_data.flatten().tolist())

        all_values = np.array(all_values)

        return {
            'min': all_values.min() if len(all_values) > 0 else 0,
            'max': all_values.max() if len(all_values) > 0 else 0,
            'mean': all_values.mean() if len(all_values) > 0 else 0,
            'median': np.median(all_values) if len(all_values) > 0 else 0
        }

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.calculate_statistics()

        print("\n" + "=" * 60)
        print("DAYLIGHT DATA STATISTICS (ALL TILES)")
        print("=" * 60)
        print(f"NPY files: {len(self.npy_files)}")
        print(f"Tiles with metadata: {len(self.metadata)}")
        print(f"\nDaylight Hours:")
        print(f"  Minimum: {stats['min']:.2f}h")
        print(f"  Maximum: {stats['max']:.2f}h")
        print(f"  Mean: {stats['mean']:.2f}h")
        print(f"  Median: {stats['median']:.2f}h")
        print("=" * 60 + "\n")


def create_maps():
    """Create daylight maps from NPY files"""

    print("=" * 60)
    print("DAYLIGHT MAP GENERATOR")
    print("=" * 60)

    # Paths
    RESULTS_DIR = "daylight_results"
    ASC_DIR = "RGEALTI/1_DONNEES_LIVRAISON_2021-10-00009/RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D074"

    # Initialize visualizer
    viz = DaylightMapVisualizer(RESULTS_DIR, ASC_DIR)

    if not viz.npy_files:
        print(f"ERROR: No NPY files found in {RESULTS_DIR}")
        print("Run compute.py first to generate daylight data")
        return

    # Print statistics
    viz.print_statistics()

    # Create lightweight debugging map
    print("\n" + "=" * 60)
    print("CREATING LIGHTWEIGHT MAP (for debugging)")
    print("=" * 60)
    print("  - 50 tiles (not all)")
    print("  - Downsampled 3x (333x333 pixels)")
    print("  - 2 markers per tile")
    print("  - Target size: ~10MB")
    viz.create_map(
        output_html="daylight_map_light.html",
        min_hours=0,
        downsample=3,
        markers_per_tile=2,
        max_tiles=50
    )

    # Create full resolution map
    print("\n" + "=" * 60)
    print("CREATING FULL RESOLUTION MAP")
    print("=" * 60)
    print("  - All tiles")
    print("  - Full resolution (1000x1000 pixels)")
    print("  - 5 markers per tile")
    viz.create_map(
        output_html="daylight_map.html",
        min_hours=0,
        downsample=1,
        markers_per_tile=5,
        max_tiles=None
    )

    print("\n" + "=" * 60)
    print("MAP GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - daylight_map_light.html (~10MB, for quick review)")
    print("  - daylight_map.html (~60MB, full resolution)")
    print("\nFeatures:")
    print("  - Corrected coordinate transformations")
    print("  - Proper pixel center handling")
    print("  - Improved marker spacing (window_size=50)")
    print("  - Multiple base layers")
    print("  - Heatmap density layer")
    print("=" * 60)


if __name__ == "__main__":
    create_maps()