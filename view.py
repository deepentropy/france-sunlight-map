import pandas as pd
import numpy as np
import os
import folium
from folium import plugins
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize


class SunlightMapVisualizer:
    """Create interactive maps from sunlight CSV data"""

    def __init__(self, csv_path="data/daylight.csv"):
        self.csv_path = csv_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Load sunlight data from CSV"""
        print(f"Loading data from {self.csv_path}...")
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.data)} data points")

            # Convert daylength_hours from timedelta string to numeric hours
            if 'daylength_hours' in self.data.columns:
                # Parse timedelta strings to actual hours
                def parse_timedelta_to_hours(td_str):
                    try:
                        # Handle format like "0 days 08:22:44.000000000"
                        if isinstance(td_str, str):
                            if 'days' in td_str:
                                parts = td_str.split()
                                days = int(parts[0])
                                time_part = parts[2]
                            else:
                                days = 0
                                time_part = td_str

                            time_parts = time_part.split(':')
                            hours = days * 24 + int(time_parts[0]) + int(time_parts[1])/60 + float(time_parts[2])/3600
                            return hours
                        else:
                            return float(td_str)
                    except:
                        return 0.0

                self.data['daylength_hours'] = self.data['daylength_hours'].apply(parse_timedelta_to_hours)

            elif 'daylength' in self.data.columns:
                # Convert daylength (seconds) to hours if needed
                self.data['daylength_hours'] = self.data['daylength'] / 3600.0

            print(f"Daylight hours range: {self.data['daylength_hours'].min():.2f}h to {self.data['daylength_hours'].max():.2f}h")
            print(f"Coordinates range:")
            print(f"  Latitude: {self.data['latitude'].min():.4f} to {self.data['latitude'].max():.4f}")
            print(f"  Longitude: {self.data['longitude'].min():.4f} to {self.data['longitude'].max():.4f}")

            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_color_for_hours(self, hours, min_hours=0, max_hours=16):
        """Get color for a given number of daylight hours"""
        # Normalize hours to 0-1 range
        norm_value = (hours - min_hours) / (max_hours - min_hours)
        norm_value = max(0, min(1, norm_value))  # Clamp to 0-1

        # Custom colormap: dark blue -> blue -> yellow -> orange -> red
        colors = ['#2c3e50', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        cmap = LinearSegmentedColormap.from_list('daylight', colors)
        rgba = cmap(norm_value)

        # Convert to hex
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    def create_comprehensive_map(self, output_html="sunlight_map_all_points.html",
                                  sample_rate=1, min_hours=0):
        """Create map with all data points"""

        if self.data is None or len(self.data) == 0:
            print("No data to visualize")
            return

        # Filter data by minimum hours
        filtered_data = self.data[self.data['daylength_hours'] >= min_hours].copy()
        print(f"Showing {len(filtered_data)} points with >= {min_hours} hours of daylight")

        # Sample data if needed for performance
        if sample_rate < 1:
            filtered_data = filtered_data.sample(frac=sample_rate, random_state=42)
            print(f"Sampled to {len(filtered_data)} points")

        # Calculate map center
        center_lat = filtered_data['latitude'].mean()
        center_lon = filtered_data['longitude'].mean()

        print(f"Map center: {center_lat:.4f}°N, {center_lon:.4f}°E")

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            max_zoom=18,
            control_scale=True,
            tiles='OpenStreetMap'
        )

        # Add multiple base layers
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

        # Add heatmap layer
        print("Creating heatmap layer...")
        self.add_heatmap_layer(m, filtered_data)

        # Add marker clusters for top locations
        print("Adding markers for top locations...")
        self.add_top_location_markers(m, filtered_data, top_n=100)

        # Add circle markers layer (optional, can be toggled)
        print("Adding circle markers...")
        self.add_circle_markers(m, filtered_data, sample_points=min(5000, len(filtered_data)))

        # Get stats for legend
        min_daylight = filtered_data['daylength_hours'].min()
        max_daylight = filtered_data['daylength_hours'].max()
        mean_daylight = filtered_data['daylength_hours'].mean()

        # Add custom legend
        legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 280px;
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0;"><b>Daylight Hours Map</b></p>
        <div style="margin: 5px 0;">
            <div style="background: linear-gradient(to right, #2c3e50, #3498db, #f1c40f, #e67e22, #e74c3c);
                        height: 20px; border-radius: 3px;"></div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 11px;">
                <span>{min_daylight:.1f}h</span>
                <span>{mean_daylight:.1f}h</span>
                <span>{max_daylight:.1f}h</span>
            </div>
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 12px;"><b>Data Points:</b> {len(filtered_data):,}</p>
        <p style="margin: 5px 0; font-size: 11px;"><small>Toggle layers in top-left panel</small></p>
        <p style="margin: 5px 0; font-size: 11px;"><small>Zoom in for street-level detail</small></p>
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

        # Save map
        m.save(output_html)
        print(f"\n{'='*60}")
        print(f"Map saved to: {output_html}")
        print(f"Total points displayed: {len(filtered_data):,}")
        print(f"Daylight range: {min_daylight:.2f}h - {max_daylight:.2f}h")
        print(f"{'='*60}\n")

        return m

    def add_heatmap_layer(self, map_obj, data):
        """Add a heatmap layer with all data points"""
        # Prepare heatmap data
        heat_data = []

        # Normalize weights based on daylight hours
        max_hours = data['daylength_hours'].max()
        min_hours = data['daylength_hours'].min()

        for _, row in data.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            hours = row['daylength_hours']

            # Normalize weight to 0-1 range
            weight = (hours - min_hours) / (max_hours - min_hours) if max_hours > min_hours else 0.5

            heat_data.append([lat, lon, weight])

        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            name='Daylight Heatmap',
            min_opacity=0.4,
            max_zoom=18,
            radius=15,
            blur=20,
            gradient={
                0.0: '#2c3e50',
                0.3: '#3498db',
                0.5: '#f1c40f',
                0.7: '#e67e22',
                1.0: '#e74c3c'
            },
            overlay=True,
            control=True,
            show=True
        ).add_to(map_obj)

    def add_top_location_markers(self, map_obj, data, top_n=100):
        """Add markers for top sunlight locations"""
        # Get top locations
        top_locations = data.nlargest(top_n, 'daylength_hours')

        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(name='Top Sunlight Locations', show=True)

        for _, row in top_locations.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            hours = row['daylength_hours']
            altitude = row.get('altitude', 'N/A')

            # Create popup with detailed info
            popup_html = f"""
            <div style="width: 220px; font-family: Arial;">
                <h4 style="margin: 5px 0; color: #e67e22;">Optimal Sunlight Location</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0;"><b>Daylight Hours:</b> {hours:.2f}h</p>
                <p style="margin: 3px 0;"><b>Altitude:</b> {altitude}m</p>
                <p style="margin: 3px 0;"><b>Coordinates:</b></p>
                <p style="margin: 3px 0; font-size: 11px;">{lat:.6f}°N, {lon:.6f}°E</p>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0; font-size: 10px; color: #666;">Click to copy coordinates</p>
            </div>
            """

            # Color based on hours
            if hours >= 8.5:
                icon_color = 'red'
            elif hours >= 8.0:
                icon_color = 'orange'
            elif hours >= 7.5:
                icon_color = 'lightred'
            else:
                icon_color = 'beige'

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{hours:.2f}h daylight",
                icon=folium.Icon(color=icon_color, icon='sun', prefix='fa')
            ).add_to(marker_cluster)

        marker_cluster.add_to(map_obj)

    def add_circle_markers(self, map_obj, data, sample_points=5000):
        """Add circle markers layer (for detailed view)"""
        # Sample data for performance
        if len(data) > sample_points:
            sampled_data = data.sample(n=sample_points, random_state=42)
        else:
            sampled_data = data

        # Create feature group
        circle_group = folium.FeatureGroup(name='Data Points (Sample)', show=False)

        max_hours = data['daylength_hours'].max()
        min_hours = data['daylength_hours'].min()

        for _, row in sampled_data.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            hours = row['daylength_hours']

            # Get color
            color = self.get_color_for_hours(hours, min_hours, max_hours)

            # Add circle marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=f"Daylight: {hours:.2f}h<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(circle_group)

        circle_group.add_to(map_obj)

    def create_statistics_report(self):
        """Generate statistics about the data"""
        if self.data is None:
            return

        print("\n" + "="*60)
        print("SUNLIGHT DATA STATISTICS")
        print("="*60)

        print(f"\nTotal data points: {len(self.data):,}")
        print(f"\nDaylight Hours:")
        print(f"  Minimum: {self.data['daylength_hours'].min():.2f}h")
        print(f"  Maximum: {self.data['daylength_hours'].max():.2f}h")
        print(f"  Mean: {self.data['daylength_hours'].mean():.2f}h")
        print(f"  Median: {self.data['daylength_hours'].median():.2f}h")

        if 'altitude' in self.data.columns:
            print(f"\nAltitude:")
            print(f"  Minimum: {self.data['altitude'].min():.0f}m")
            print(f"  Maximum: {self.data['altitude'].max():.0f}m")
            print(f"  Mean: {self.data['altitude'].mean():.0f}m")

        print(f"\nTop 10 locations with most daylight:")
        top_10 = self.data.nlargest(10, 'daylength_hours')[['latitude', 'longitude', 'daylength_hours', 'altitude']]
        for idx, row in top_10.iterrows():
            print(f"  {row['daylength_hours']:.2f}h at ({row['latitude']:.4f}, {row['longitude']:.4f}) - {row['altitude']:.0f}m")

        print("="*60 + "\n")


def create_all_maps():
    """Create comprehensive sunlight maps from CSV data"""

    print("=" * 60)
    print("SUNLIGHT MAP GENERATOR")
    print("=" * 60)

    # Initialize visualizer
    viz = SunlightMapVisualizer(csv_path="data/daylight.csv")

    if viz.data is None:
        print("Error: Could not load data")
        return

    # Print statistics
    viz.create_statistics_report()

    # Create comprehensive map with all points
    print("\nCreating comprehensive map with all data points...")
    viz.create_comprehensive_map(
        output_html="sunlight_map_complete.html",
        sample_rate=1.0,  # Use all points
        min_hours=0  # Show all data
    )

    # Create map showing only high-sunlight areas
    print("\nCreating map for high-sunlight areas (>7.5h)...")
    viz.create_comprehensive_map(
        output_html="sunlight_map_high_areas.html",
        sample_rate=1.0,
        min_hours=7.5
    )

    print("\n" + "=" * 60)
    print("MAP GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. sunlight_map_complete.html - All data points")
    print("  2. sunlight_map_high_areas.html - High sunlight areas only")
    print("\nFeatures:")
    print("  - Interactive heatmap layer")
    print("  - Top 100 location markers")
    print("  - Multiple base layers (Street, Satellite, Topographic)")
    print("  - Measurement tools")
    print("  - Fullscreen mode")
    print("  - Layer toggle controls")
    print("="*60)


if __name__ == "__main__":
    create_all_maps()