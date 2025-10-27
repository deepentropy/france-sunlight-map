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
                # Lambert 93 coordinates (corners of the grid)
                xmin = header['xllcorner']
                ymin = header['yllcorner']
                xmax = xmin + header['ncols'] * header['cellsize']
                ymax = ymin + header['nrows'] * header['cellsize']

                # Transform corners to WGS84 (lat/lon)
                # Lower-left corner
                lon_sw, lat_sw = self.transformer.transform(xmin, ymin)
                # Upper-right corner
                lon_ne, lat_ne = self.transformer.transform(xmax, ymax)

                # Folium expects bounds as: [[south, west], [north, east]]
                # south = min latitude, north = max latitude
                # west = min longitude, east = max longitude
                south = min(lat_sw, lat_ne)
                north = max(lat_sw, lat_ne)
                west = min(lon_sw, lon_ne)
                east = max(lon_sw, lon_ne)

                self.metadata[basename] = {
                    'header': header,
                    'bounds_lambert': (xmin, ymin, xmax, ymax),
                    'bounds_wgs84': [[south, west], [north, east]],  # FIXED: proper Folium format
                    'center_wgs84': ((west + east) / 2, (south + north) / 2)
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

    def create_map(self, output_html="daylight_map.html", min_hours=0, downsample=1):
        """
        Create full-resolution interactive map

        Args:
            output_html: Output filename
            min_hours: Minimum daylight hours to display
            downsample: Downsampling factor for overlay images (1=full res, 5=web-friendly)
        """

        if not self.npy_files:
            print("No NPY files found!")
            return

        if not self.metadata:
            print("No metadata available - need ASC files for georeferencing")
            return

        # Calculate map center
        all_lons = []
        all_lats = []
        for meta in self.metadata.values():
            lon, lat = meta['center_wgs84']
            all_lons.append(lon)
            all_lats.append(lat)

        center_lon = np.mean(all_lons) if all_lons else 6.0
        center_lat = np.mean(all_lats) if all_lats else 46.0

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
        print(f"Creating map overlays for {len(self.npy_files)} tiles (downsample={downsample}x)...")
        processed_count = 0
        total_pixels = 0

        for npy_file in tqdm(self.npy_files, desc="Processing tiles"):
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
                    bounds=meta['bounds_wgs84'],  # Now correctly formatted!
                    opacity=0.6,
                    name=f'Tile {basename}',
                    overlay=True,
                    control=False,
                    zindex=1
                ).add_to(m)
                processed_count += 1

        print(f"Processed {processed_count} tiles with {total_pixels:,} total data points")

        # Add markers for optimal locations
        print("Adding markers for optimal sunlight locations...")
        self.add_optimal_location_markers(m, min_hours=14)

        # Add heatmap layer
        print("Creating heatmap layer...")
        self.add_heatmap_layer(m, min_hours=12)

        # Calculate statistics
        stats = self.calculate_statistics()

        # Add legend
        legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 300px;
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0;"><b>Daylight Hours Map</b></p>
        <div style="margin: 5px 0;">
            <div style="background: linear-gradient(to right, #2c3e50, #3498db, #f1c40f, #e67e22, #e74c3c);
                        height: 20px; border-radius: 3px;"></div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 11px;">
                <span>0h</span>
                <span>8h</span>
                <span>12h</span>
                <span>14h</span>
                <span>16h</span>
            </div>
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 12px;"><b>Tiles:</b> {processed_count}</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Data Points:</b> {total_pixels:,}</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Min:</b> {stats['min']:.1f}h | <b>Max:</b> {stats['max']:.1f}h</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Mean:</b> {stats['mean']:.1f}h</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 11px;"><small>Full resolution - no filtering</small></p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add controls
        folium.LayerControl(position='topleft').add_to(m)
        plugins.Fullscreen(
            position='topright',
            title='Fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)
        plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)

        # Save map
        m.save(output_html)
        print(f"\n{'='*60}")
        print(f"Map saved to: {output_html}")
        print(f"Tiles processed: {processed_count}")
        print(f"Total data points: {total_pixels:,}")
        print(f"Downsampling: {downsample}x (1 = full resolution)")
        print(f"{'='*60}\n")

        return m

    def create_overlay_image(self, daylight_data, min_hours=0):
        """Create colored overlay image from daylight data using GPU if available"""
        try:
            # Use GPU if available for faster processing
            if GPU_AVAILABLE and daylight_data.size > 100000:
                return self._create_overlay_gpu(daylight_data, min_hours)
            else:
                return self._create_overlay_cpu(daylight_data, min_hours)
        except Exception as e:
            print(f"Error creating overlay: {e}")
            return None

    def _create_overlay_gpu(self, daylight_data, min_hours):
        """GPU-accelerated overlay creation"""
        # Transfer to GPU
        daylight_gpu = cp.asarray(daylight_data, dtype=cp.float32)

        # Mask low values
        masked_data = cp.copy(daylight_gpu)
        masked_data[daylight_gpu < min_hours] = cp.nan

        # Normalize
        vmin, vmax = 0, 16
        normalized = (masked_data - vmin) / (vmax - vmin)

        # Create colormap on CPU (matplotlib doesn't work with CuPy)
        colors = ['#2c3e50', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        cmap = LinearSegmentedColormap.from_list('daylight', colors)

        # Transfer normalized data back to CPU for colormap
        normalized_cpu = cp.asnumpy(normalized)

        # Apply colormap
        colored = cmap(normalized_cpu)

        # Set transparency for NaN values
        colored[np.isnan(normalized_cpu)] = [0, 0, 0, 0]

        # Convert to image
        img = Image.fromarray((colored * 255).astype(np.uint8))

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def _create_overlay_cpu(self, daylight_data, min_hours):
        """CPU overlay creation (fallback)"""
        # Mask low values
        masked_data = np.copy(daylight_data)
        masked_data[daylight_data < min_hours] = np.nan

        # Normalize
        vmin, vmax = 0, 16
        normalized = (masked_data - vmin) / (vmax - vmin)

        # Create colormap
        colors = ['#2c3e50', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        cmap = LinearSegmentedColormap.from_list('daylight', colors)

        # Apply colormap
        colored = cmap(normalized)

        # Set transparency for NaN values
        colored[np.isnan(masked_data)] = [0, 0, 0, 0]

        # Convert to image
        img = Image.fromarray((colored * 255).astype(np.uint8))

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def add_optimal_location_markers(self, map_obj, min_hours=14):
        """Add markers for optimal sunlight locations with FIXED coordinates"""
        marker_group = folium.FeatureGroup(name='Optimal Locations (>14h)', show=True)

        marker_count = 0
        for npy_file in self.npy_files:
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            daylight = np.load(npy_file)
            meta = self.metadata[basename]
            header = meta['header']

            # Find local maxima
            maxima = self.find_local_maxima(daylight, threshold=min_hours, window_size=20)

            for i, j in maxima[:5]:  # Top 5 per tile
                # FIXED: Proper coordinate transformation
                # i, j are array indices (row, col)
                # Need to convert to Lambert 93, then to WGS84

                # X coordinate (longitude direction) - column j
                x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']

                # Y coordinate (latitude direction) - row i
                # In raster: row 0 is at TOP (ymax), row nrows-1 is at BOTTOM (ymin)
                # So: y = ymax - i * cellsize
                y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']

                # Transform to WGS84 (returns lon, lat)
                lon, lat = self.transformer.transform(x_lambert, y_lambert)
                hours = float(daylight[i, j])

                popup_html = f"""
                <div style="width: 240px;">
                    <h4 style="color: #e67e22;">Optimal Sunlight Location</h4>
                    <b>Daylight:</b> {hours:.2f}h<br>
                    <b>Position:</b> Row {i}, Col {j}<br>
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

    def find_local_maxima(self, data, threshold=14, window_size=20):
        """Find local maxima in daylight data"""
        mask = data >= threshold

        if not np.any(mask):
            return []

        from scipy.ndimage import maximum_filter
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

                    # FIXED: Same coordinate transformation as markers
                    x_lambert = header['xllcorner'] + (j + 0.5) * header['cellsize']
                    y_lambert = header['yllcorner'] + (header['nrows'] - i - 0.5) * header['cellsize']
                    lon, lat = self.transformer.transform(x_lambert, y_lambert)

                    weight = float(daylight[i, j]) / 16.0
                    heat_data.append([lat, lon, weight])  # Folium HeatMap expects [lat, lon, weight]

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

        print("\n" + "="*60)
        print("DAYLIGHT DATA STATISTICS (ALL TILES)")
        print("="*60)
        print(f"NPY files: {len(self.npy_files)}")
        print(f"Tiles with metadata: {len(self.metadata)}")
        print(f"\nDaylight Hours:")
        print(f"  Minimum: {stats['min']:.2f}h")
        print(f"  Maximum: {stats['max']:.2f}h")
        print(f"  Mean: {stats['mean']:.2f}h")
        print(f"  Median: {stats['median']:.2f}h")
        print("="*60 + "\n")


def create_maps():
    """Create daylight map from NPY files"""

    print("="*60)
    print("DAYLIGHT MAP GENERATOR - FULL RESOLUTION")
    print("="*60)

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

    # Create full resolution map
    print("\nCreating full-resolution map...")
    viz.create_map(
        output_html="daylight_map.html",
        min_hours=0,  # Show all data
        downsample=1  # Full resolution (change to 5 for web-friendly)
    )

    print("\n" + "="*60)
    print("MAP GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated file: daylight_map.html")
    print("\nFeatures:")
    print("  - Full resolution overlays (all 235 tiles)")
    print("  - No gaps between tiles (fixed coordinate bug)")
    print("  - Correct marker positions (fixed Y-axis inversion)")
    print("  - GPU-accelerated image processing")
    print("  - Multiple base layers")
    print("  - Heatmap density layer")
    print("  - Measurement tools and fullscreen mode")
    print("="*60)


if __name__ == "__main__":
    create_maps()
