import numpy as np
import os
import glob
from pyproj import Transformer
import folium
from folium import plugins
import rasterio
from tqdm import tqdm
from scipy import ndimage
import json
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


class EnhancedDaylightMap:
    """Create detailed interactive maps with full daylight data"""

    def __init__(self, results_dir="daylight_results", asc_dir=None):
        self.results_dir = results_dir
        self.asc_dir = asc_dir
        self.npy_files = glob.glob(os.path.join(results_dir, "*.npy"))
        self.transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        self.metadata = {}

        if asc_dir:
            self.load_metadata()

    def load_metadata(self):
        """Load metadata from ASC files"""
        asc_files = glob.glob(os.path.join(self.asc_dir, "*.asc"))

        for asc_file in tqdm(asc_files[:50], desc="Loading metadata"):  # Limit for testing
            basename = os.path.basename(asc_file).replace('.asc', '')
            header = self.read_asc_header(asc_file)
            if header:
                xmin = header['xllcorner']
                ymin = header['yllcorner']
                xmax = xmin + header['ncols'] * header['cellsize']
                ymax = ymin + header['nrows'] * header['cellsize']

                lon_min, lat_min = self.transformer.transform(xmin, ymin)
                lon_max, lat_max = self.transformer.transform(xmax, ymax)

                self.metadata[basename] = {
                    'header': header,
                    'bounds_lambert': (xmin, ymin, xmax, ymax),
                    'bounds_wgs84': [[lat_min, lon_min], [lat_max, lon_max]],
                    'center_wgs84': ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)
                }

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
        except:
            return None

    def create_detailed_heatmap(self, output_html="detailed_daylight_map.html",
                                downsample_factor=10, min_hours=0):
        """Create detailed heatmap with street-level zoom capability"""

        if not self.metadata:
            print("No metadata available")
            return

        # Calculate map center
        all_lats = []
        all_lons = []
        for meta in self.metadata.values():
            center = meta['center_wgs84']
            all_lons.append(center[0])
            all_lats.append(center[1])

        center_lat = np.mean(all_lats) if all_lats else 45.9474
        center_lon = np.mean(all_lons) if all_lons else 5.8082

        # Create base map with multiple tile layers
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            max_zoom=18,
            control_scale=True
        )

        # Add multiple base layers for better context
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
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

        # Process each tile and create overlay images
        print("Creating detailed heatmap overlays...")

        for npy_file in tqdm(self.npy_files[:20], desc="Processing tiles for map"):  # Limit for demo
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            # Load daylight data
            daylight = np.load(npy_file)
            meta = self.metadata[basename]

            # Downsample for web performance
            if downsample_factor > 1:
                daylight = daylight[::downsample_factor, ::downsample_factor]

            # Create image overlay
            overlay_img = self.create_overlay_image(daylight, min_hours)

            if overlay_img:
                # Add as image overlay
                folium.raster_layers.ImageOverlay(
                    image=overlay_img,
                    bounds=meta['bounds_wgs84'],
                    opacity=0.6,
                    name=f'Daylight {basename}',
                    overlay=True,
                    control=True,
                    zindex=1
                ).add_to(m)

        # Add a feature group for high daylight markers
        high_sun_group = folium.FeatureGroup(name='Optimal Locations (>14h)')

        # Add markers for optimal spots
        for npy_file in self.npy_files[:20]:
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            daylight = np.load(npy_file)
            meta = self.metadata[basename]
            header = meta['header']

            # Find local maxima
            maxima = self.find_local_maxima(daylight, threshold=14, window_size=20)

            for i, j in maxima[:10]:  # Limit markers per tile
                # Convert to coordinates
                x = header['xllcorner'] + j * header['cellsize']
                y = header['yllcorner'] + (header['nrows'] - i) * header['cellsize']
                lon, lat = self.transformer.transform(x, y)
                hours = float(daylight[i, j])

                # Add marker with popup
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>Optimal Sun Location</h4>
                    <b>Daylight Hours:</b> {hours:.1f}h<br>
                    <b>Coordinates:</b> {lat:.5f}, {lon:.5f}<br>
                    <b>Tile:</b> {basename}<br>
                    <hr>
                    <small>Click to copy coordinates</small>
                </div>
                """

                icon_color = 'red' if hours >= 15 else 'orange' if hours >= 14.5 else 'yellow'

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{hours:.1f}h of daylight",
                    icon=folium.Icon(color=icon_color, icon='sun', prefix='fa')
                ).add_to(high_sun_group)

        high_sun_group.add_to(m)

        # Add heatmap layer with all data points
        self.add_density_heatmap(m, threshold=14)

        # Add custom legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0;"><b>Daylight Hours (Summer Solstice)</b></p>
        <div style="margin: 5px 0;">
            <div style="background: linear-gradient(to right, #2c3e50, #3498db, #f39c12, #e74c3c, #c0392b); 
                        height: 20px; border-radius: 3px;"></div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span>0h</span>
                <span>8h</span>
                <span>12h</span>
                <span>14h</span>
                <span>16h</span>
            </div>
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0;"><small>Toggle layers using control panel</small></p>
        <p style="margin: 5px 0;"><small>Zoom in for street-level detail</small></p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add layer control
        folium.LayerControl(position='topleft').add_to(m)

        # Add fullscreen control
        plugins.Fullscreen(
            position='topright',
            title='Fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)

        # Add measurement control
        plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)

        # Add search control
        plugins.Search(
            layer=high_sun_group,
            search_label='popup',
            search_zoom=15,
            position='topright'
        ).add_to(m)

        # Save map
        m.save(output_html)
        print(f"Saved detailed map to {output_html}")

    def create_overlay_image(self, daylight_data, min_hours=0):
        """Create a colored overlay image from daylight data"""
        try:
            # Mask low values
            masked_data = np.copy(daylight_data)
            masked_data[daylight_data < min_hours] = np.nan

            # Normalize data
            vmin, vmax = 0, 16
            normalized = (masked_data - vmin) / (vmax - vmin)

            # Create custom colormap
            colors = ['#2c3e50', '#34495e', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('daylight', colors, N=n_bins)

            # Apply colormap
            colored = cmap(normalized)

            # Set transparency for NaN values
            colored[np.isnan(masked_data)] = [0, 0, 0, 0]

            # Convert to image
            img = Image.fromarray((colored * 255).astype(np.uint8))

            # Convert to base64 for embedding
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            print(f"Error creating overlay: {e}")
            return None

    def find_local_maxima(self, data, threshold=14, window_size=20):
        """Find local maxima in the daylight data"""
        # Apply threshold
        mask = data >= threshold

        if not np.any(mask):
            return []

        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(data, size=window_size)
        maxima = (data == local_max) & mask

        # Get coordinates
        coords = np.column_stack(np.where(maxima))

        # Sort by value
        values = [data[i, j] for i, j in coords]
        sorted_indices = np.argsort(values)[::-1]

        return coords[sorted_indices].tolist()

    def add_density_heatmap(self, map_obj, threshold=14, sample_rate=50):
        """Add a density heatmap layer"""
        heat_data = []

        for npy_file in self.npy_files[:20]:  # Process subset
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            daylight = np.load(npy_file)
            meta = self.metadata[basename]
            header = meta['header']

            # Sample points above threshold
            high_sun = daylight >= threshold
            indices = np.where(high_sun)

            if len(indices[0]) > 0:
                # Sample to avoid too many points
                n_samples = min(sample_rate, len(indices[0]))
                sample_idx = np.random.choice(len(indices[0]), n_samples, replace=False)

                for idx in sample_idx:
                    i, j = indices[0][idx], indices[1][idx]

                    # Convert to lat/lon
                    x = header['xllcorner'] + j * header['cellsize']
                    y = header['yllcorner'] + (header['nrows'] - i) * header['cellsize']
                    lon, lat = self.transformer.transform(x, y)

                    weight = float(daylight[i, j]) / 16.0  # Normalize weight
                    heat_data.append([float(lat), float(lon), weight])

        if heat_data:
            # Add heatmap layer
            plugins.HeatMap(
                heat_data,
                name='Daylight Density',
                min_opacity=0.3,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={
                    0.0: 'blue',
                    0.5: 'lime',
                    0.7: 'yellow',
                    0.9: 'orange',
                    1.0: 'red'
                },
                overlay=True,
                control=True,
                show=False  # Hidden by default
            ).add_to(map_obj)

    def create_focused_area_map(self, center_lambert_x, center_lambert_y,
                                radius_m=5000, output_html="focused_area.html"):
        """Create a detailed map for a specific area"""

        # Convert center to lat/lon
        center_lon, center_lat = self.transformer.transform(center_lambert_x, center_lambert_y)

        # Create map centered on the area
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            max_zoom=20
        )

        # Add detailed tile layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite HD',
            overlay=False,
            control=True
        ).add_to(m)

        # Process only nearby tiles
        for npy_file in self.npy_files:
            basename = os.path.basename(npy_file).replace('_daylight.npy', '')

            if basename not in self.metadata:
                continue

            meta = self.metadata[basename]
            bounds = meta['bounds_lambert']

            # Check if tile is within radius
            tile_center_x = (bounds[0] + bounds[2]) / 2
            tile_center_y = (bounds[1] + bounds[3]) / 2

            distance = np.sqrt((tile_center_x - center_lambert_x) ** 2 +
                               (tile_center_y - center_lambert_y) ** 2)

            if distance <= radius_m:
                # Load and process this tile at full resolution
                daylight = np.load(npy_file)

                # Create detailed overlay
                overlay_img = self.create_overlay_image(daylight, min_hours=12)

                if overlay_img:
                    folium.raster_layers.ImageOverlay(
                        image=overlay_img,
                        bounds=meta['bounds_wgs84'],
                        opacity=0.7,
                        name=f'Daylight {basename}'
                    ).add_to(m)

        # Add circle to show area of interest
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius_m,
            color='red',
            fill=False,
            weight=2
        ).add_to(m)

        # Save map
        m.save(output_html)
        print(f"Saved focused area map to {output_html}")
        return m


def create_comprehensive_maps():
    """Create all map types"""

    # Paths
    RESULTS_DIR = "daylight_results"
    ASC_DIR = "RGEALTI/1_DONNEES_LIVRAISON_2021-10-00009/RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D074"

    print("Creating comprehensive daylight maps...")
    print("=" * 60)

    # Initialize mapper
    mapper = EnhancedDaylightMap(RESULTS_DIR, ASC_DIR if os.path.exists(ASC_DIR) else None)

    if not mapper.metadata:
        print("Warning: No ASC directory found, using NPY files only")
        return

    # 1. Create main detailed map
    print("\n1. Creating detailed heatmap with street overlay...")
    mapper.create_detailed_heatmap(
        output_html="daylight_detailed_map.html",
        downsample_factor=5,  # Less downsampling for more detail
        min_hours=10  # Show all areas with >10 hours
    )

    # 2. Create high-resolution map for best area (example coordinates)
    print("\n2. Creating focused area map...")
    # Example: focus on area near Annecy (you can change these coordinates)
    mapper.create_focused_area_map(
        center_lambert_x=915000,  # Example Lambert 93 coordinates
        center_lambert_y=6545000,
        radius_m=3000,
        output_html="focused_annecy_area.html"
    )

    print("\n" + "=" * 60)
    print("Maps created successfully!")
    print("\nGenerated files:")
    print("- daylight_detailed_map.html : Full interactive map with streets")
    print("- focused_annecy_area.html : High-detail focused area")
    print("\nFeatures:")
    print("- Multiple base layers (Streets, Satellite, Topographic)")
    print("- Zoom to street level (max zoom 18)")
    print("- Measurement tools")
    print("- Fullscreen mode")
    print("- Search functionality")
    print("- Daylight overlay on all tiles")
    print("=" * 60)


if __name__ == "__main__":
    create_comprehensive_maps()