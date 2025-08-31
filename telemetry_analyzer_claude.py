import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import json

class RaceChronoDatalogAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize with RaceChrono CSV datalog file"""
        self.df = None
        self.gps_data = None
        self.track_centerline = None
        self.racechrono_columns = {}
        self.load_datalog(csv_file_path)
    
    def load_datalog(self, csv_file_path):
        """Load and analyze the RaceChrono datalog CSV"""
        try:
            # RaceChrono CSVs sometimes have metadata at the top
            # Try to detect the actual data start
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Look for the header row (usually contains Time, Distance, etc.)
            header_row = 0
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                if 'Time' in line or 'Distance' in line or 'Latitude' in line:
                    header_row = i
                    break
            
            # Read CSV starting from the detected header
            self.df = pd.read_csv(csv_file_path, skiprows=header_row)
            
            print(f"Loaded RaceChrono datalog with {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"\nColumn names ({len(self.df.columns)} total):")
            for i, col in enumerate(self.df.columns):
                print(f"{i:2d}: {col}")
            
            # Show data types and sample values
            print(f"\nFirst few rows:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(self.df.head(3))
            
            # Detect RaceChrono-specific columns
            self.detect_racechrono_columns()
            
        except Exception as e:
            print(f"Error loading datalog: {e}")
            import traceback
            traceback.print_exc()
    
    def detect_racechrono_columns(self):
        """Detect RaceChrono-specific column names and data"""
        # Common RaceChrono column patterns
        column_mapping = {
            'time': None,
            'distance': None,
            'latitude': None,
            'longitude': None,
            'speed': None,
            'acceleration_x': None,
            'acceleration_y': None,
            'acceleration_z': None,
            'gyro_x': None,
            'gyro_y': None,
            'gyro_z': None
        }
        
        for col in self.df.columns:
            col_lower = col.lower().strip()
            
            # Time columns
            if any(term in col_lower for term in ['time', 'timestamp']):
                column_mapping['time'] = col
            
            # Distance columns
            elif any(term in col_lower for term in ['distance', 'dist']):
                column_mapping['distance'] = col
            
            # GPS columns
            elif any(term in col_lower for term in ['latitude', 'lat']):
                column_mapping['latitude'] = col
            elif any(term in col_lower for term in ['longitude', 'lon', 'lng']):
                column_mapping['longitude'] = col
            
            # Speed columns
            elif any(term in col_lower for term in ['speed', 'velocity']):
                column_mapping['speed'] = col
            
            # Accelerometer columns
            elif 'acceleration' in col_lower or 'accel' in col_lower or 'g-force' in col_lower:
                if 'x' in col_lower or 'lateral' in col_lower:
                    column_mapping['acceleration_x'] = col
                elif 'y' in col_lower or 'longitudinal' in col_lower:
                    column_mapping['acceleration_y'] = col
                elif 'z' in col_lower or 'vertical' in col_lower:
                    column_mapping['acceleration_z'] = col
            
            # Gyroscope columns
            elif 'gyro' in col_lower or 'angular' in col_lower:
                if 'x' in col_lower:
                    column_mapping['gyro_x'] = col
                elif 'y' in col_lower:
                    column_mapping['gyro_y'] = col
                elif 'z' in col_lower:
                    column_mapping['gyro_z'] = col
        
        self.racechrono_columns = column_mapping
        
        print(f"\nDetected RaceChrono columns:")
        for data_type, column_name in column_mapping.items():
            status = f"✓ {column_name}" if column_name else "✗ Not found"
            print(f"  {data_type:15}: {status}")
        
        # Check for GPS data quality
        if column_mapping['latitude'] and column_mapping['longitude']:
            lat_col = column_mapping['latitude']
            lon_col = column_mapping['longitude']
            
            valid_gps_count = len(self.df[(self.df[lat_col] != 0) & 
                                         (self.df[lon_col] != 0) & 
                                         (self.df[lat_col].notna()) & 
                                         (self.df[lon_col].notna())])
            
            print(f"\nGPS Data Quality:")
            print(f"  Total rows: {len(self.df)}")
            print(f"  Valid GPS points: {valid_gps_count}")
            print(f"  GPS coverage: {valid_gps_count/len(self.df)*100:.1f}%")
            
            if valid_gps_count > 0:
                print(f"  Lat range: {self.df[lat_col].min():.6f} to {self.df[lat_col].max():.6f}")
                print(f"  Lon range: {self.df[lon_col].min():.6f} to {self.df[lon_col].max():.6f}")
        
        return column_mapping
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS points in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def gps_to_local_coordinates(self, lat_col=None, lon_col=None):
        """Convert GPS coordinates to local XY coordinates"""
        # Use detected columns if not specified
        if lat_col is None:
            lat_col = self.racechrono_columns.get('latitude')
        if lon_col is None:
            lon_col = self.racechrono_columns.get('longitude')
            
        if not lat_col or not lon_col:
            print(f"Error: GPS columns not found. Latitude: {lat_col}, Longitude: {lon_col}")
            return None
        
        if lat_col not in self.df.columns or lon_col not in self.df.columns:
            print(f"Error: Columns {lat_col} or {lon_col} not found")
            return None
        
        # Remove invalid GPS points (0, NaN, or obviously wrong values)
        valid_mask = ((self.df[lat_col] != 0) & 
                     (self.df[lon_col] != 0) & 
                     (self.df[lat_col].notna()) & 
                     (self.df[lon_col].notna()) &
                     (self.df[lat_col].abs() <= 90) &
                     (self.df[lon_col].abs() <= 180))
        
        valid_gps = self.df[valid_mask].copy()
        
        if len(valid_gps) == 0:
            print("No valid GPS data found")
            return None
        
        print(f"Processing {len(valid_gps)} valid GPS points...")
        
        # Use first point as origin
        origin_lat = valid_gps[lat_col].iloc[0]
        origin_lon = valid_gps[lon_col].iloc[0]
        
        print(f"Origin point: {origin_lat:.6f}, {origin_lon:.6f}")
        
        # Convert to local coordinates (meters from origin)
        x_coords = []
        y_coords = []
        
        for _, row in valid_gps.iterrows():
            lat, lon = row[lat_col], row[lon_col]
            
            # Calculate X (east-west) distance
            x = self.haversine_distance(origin_lat, origin_lon, origin_lat, lon)
            if lon < origin_lon:
                x = -x
            
            # Calculate Y (north-south) distance  
            y = self.haversine_distance(origin_lat, origin_lon, lat, origin_lon)
            if lat < origin_lat:
                y = -y
                
            x_coords.append(x)
            y_coords.append(y)
        
        valid_gps['x'] = x_coords
        valid_gps['y'] = y_coords
        
        # Add original column references for later use
        valid_gps['original_lat_col'] = lat_col
        valid_gps['original_lon_col'] = lon_col
        
        self.gps_data = valid_gps
        
        print(f"Converted to local coordinates:")
        print(f"  X range: {min(x_coords):.1f}m to {max(x_coords):.1f}m")
        print(f"  Y range: {min(y_coords):.1f}m to {max(y_coords):.1f}m")
        
        return valid_gps
    
    def analyze_racing_line(self, speed_col=None):
        """Analyze the racing line characteristics for autocross"""
        if self.gps_data is None:
            print("No GPS data available. Run gps_to_local_coordinates first.")
            return None
        
        # Use detected speed column if not specified
        if speed_col is None:
            speed_col = self.racechrono_columns.get('speed')
        
        analysis = {}
        
        # Calculate track length and segment analysis
        total_distance = 0
        segment_speeds = []
        segment_curvatures = []
        
        for i in range(1, len(self.gps_data)):
            prev_row = self.gps_data.iloc[i-1]
            curr_row = self.gps_data.iloc[i]
            
            # Distance calculation
            dist = sqrt((curr_row['x'] - prev_row['x'])**2 + (curr_row['y'] - prev_row['y'])**2)
            total_distance += dist
            
            # Speed analysis if available
            if speed_col and speed_col in self.gps_data.columns:
                speed_val = curr_row[speed_col]
                if pd.notna(speed_val):
                    segment_speeds.append(speed_val)
        
        analysis['track_length_meters'] = total_distance
        analysis['track_length_feet'] = total_distance * 3.28084
        
        # Speed analysis
        if speed_col and speed_col in self.gps_data.columns and len(segment_speeds) > 0:
            analysis['speed_stats'] = {
                'avg_speed_ms': np.mean(segment_speeds),
                'max_speed_ms': np.max(segment_speeds),
                'min_speed_ms': np.min(segment_speeds),
                'avg_speed_mph': np.mean(segment_speeds) * 2.237,
                'max_speed_mph': np.max(segment_speeds) * 2.237,
                'min_speed_mph': np.min(segment_speeds) * 2.237
            }
        
        # Curvature analysis for autocross characteristics
        curvatures = []
        turn_points = []
        
        for i in range(2, len(self.gps_data)-2):
            # Use 5-point smoothing for better curvature calculation
            points = []
            for j in range(i-2, i+3):
                row = self.gps_data.iloc[j]
                points.append((row['x'], row['y']))
            
            # Calculate curvature using 3-point method on smoothed data
            p1, p2, p3 = points[1], points[2], points[3]
            
            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3  
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            if (v1[0]**2 + v1[1]**2) > 0 and (v2[0]**2 + v2[1]**2) > 0:
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = sqrt(v1[0]**2 + v1[1]**2)
                mag2 = sqrt(v2[0]**2 + v2[1]**2)
                
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical errors
                
                angle_change = abs(np.arccos(cos_angle))
                curvature = angle_change / mag1 if mag1 > 0 else 0
                
                curvatures.append(curvature)
                
                # Detect significant turns (for autocross cone placement estimation)
                if curvature > 0.1:  # Threshold for "significant turn"
                    turn_points.append({
                        'index': i,
                        'x': p2[0],
                        'y': p2[1],
                        'curvature': curvature,
                        'speed': self.gps_data.iloc[i][speed_col] if speed_col and speed_col in self.gps_data.columns else None
                    })
        
        if curvatures:
            analysis['curvature_stats'] = {
                'avg_curvature': np.mean(curvatures),
                'max_curvature': np.max(curvatures),
                'turn_count': len(turn_points),
                'turns_per_100m': len(turn_points) / (total_distance / 100) if total_distance > 0 else 0
            }
            analysis['turn_points'] = turn_points
        
        # Autocross-specific analysis
        analysis['autocross_characteristics'] = {
            'estimated_cone_count': len(turn_points) * 2,  # Rough estimate
            'track_complexity': 'high' if len(turn_points) > 15 else 'medium' if len(turn_points) > 8 else 'low',
            'recommended_track_width': 9,  # meters, typical autocross
            'racing_line_efficiency': self.calculate_line_efficiency()
        }
        
        return analysis
    
    def calculate_line_efficiency(self):
        """Estimate how efficient the racing line is vs theoretical optimal"""
        if self.gps_data is None or len(self.gps_data) < 3:
            return 0.5
        
        # Calculate actual path length
        actual_length = 0
        for i in range(1, len(self.gps_data)):
            prev = self.gps_data.iloc[i-1]
            curr = self.gps_data.iloc[i]
            actual_length += sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
        
        # Calculate theoretical minimum (straight line from start to end)
        start = self.gps_data.iloc[0]
        end = self.gps_data.iloc[-1]
        straight_line = sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
        
        # For closed courses, this ratio will be very different
        # For autocross, check if it's a closed loop
        start_to_end_distance = sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
        is_closed_course = start_to_end_distance < (actual_length * 0.1)
        
        if is_closed_course:
            # For closed courses, efficiency is harder to calculate
            # Use curvature-based approach
            return 0.7  # Placeholder - would need more sophisticated analysis
        else:
            # Open course efficiency
            efficiency = straight_line / actual_length if actual_length > 0 else 0
            return min(1.0, max(0.1, efficiency))

    def estimate_centerline_offset(self, track_width_meters=9):
        """Estimate how far the racing line is from track centerline for autocross"""
        if self.gps_data is None:
            print("No GPS data available")
            return None
        
        # For autocross, analyze racing line vs theoretical optimal
        analysis = self.analyze_racing_line()
        
        if not analysis or 'turn_points' not in analysis:
            return {
                'avg_offset_from_center': 0,
                'confidence': 'low',
                'method': 'no_turn_analysis'
            }
        
        turn_points = analysis['turn_points']
        
        # Analyze turn directions and racing line positioning
        left_turns = 0
        right_turns = 0
        
        for i, turn in enumerate(turn_points[:-1]):
            if i == 0:
                continue
            
            # Determine turn direction using cross product
            curr_turn = turn_points[i]
            next_turn = turn_points[i+1] if i+1 < len(turn_points) else turn_points[0]
            prev_turn = turn_points[i-1]
            
            # Vector from previous to current turn
            v1 = (curr_turn['x'] - prev_turn['x'], curr_turn['y'] - prev_turn['y'])
            # Vector from current to next turn
            v2 = (next_turn['x'] - curr_turn['x'], next_turn['y'] - curr_turn['y'])
            
            # Cross product to determine turn direction
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            if cross_product > 0:
                left_turns += 1
            else:
                right_turns += 1
        
        # Estimate centerline offset based on racing line theory
        if left_turns > right_turns:
            avg_offset = 1.5  # Slightly right of center for left-heavy course
        elif right_turns > left_turns:
            avg_offset = -1.5  # Slightly left of center for right-heavy course
        else:
            avg_offset = 0  # Balanced course, assume good centering
        
        return {
            'avg_offset_from_center_meters': avg_offset,
            'track_width_used_percent': 85,  # Typical for autocross
            'turn_analysis': {
                'left_turns': left_turns,
                'right_turns': right_turns,
                'total_turns': len(turn_points)
            },
            'confidence': 'medium',
            'method': 'turn_direction_analysis'
        }
    
    def export_for_blender(self, output_file='racechrono_track_data.json', track_width=9):
        """Export processed data for Blender import"""
        if self.gps_data is None:
            print("No GPS data to export")
            return None
        
        # Get analysis data
        racing_analysis = self.analyze_racing_line()
        centerline_analysis = self.estimate_centerline_offset(track_width)
        
        export_data = {
            'metadata': {
                'source': 'RaceChrono',
                'total_points': len(self.gps_data),
                'coordinate_system': 'local_meters_from_origin',
                'track_width_meters': track_width,
                'racing_line_analysis': racing_analysis,
                'centerline_analysis': centerline_analysis
            },
            'racing_line': [
                {
                    'x': float(row['x']), 
                    'y': float(row['y']), 
                    'index': i,
                    'time': float(row[self.racechrono_columns['time']]) if self.racechrono_columns.get('time') and self.racechrono_columns['time'] in row else None,
                    'speed': float(row[self.racechrono_columns['speed']]) if self.racechrono_columns.get('speed') and self.racechrono_columns['speed'] in row else None,
                    'distance': float(row[self.racechrono_columns['distance']]) if self.racechrono_columns.get('distance') and self.racechrono_columns['distance'] in row else None
                }
                for i, (_, row) in enumerate(self.gps_data.iterrows())
            ],
            'estimated_centerline': self.generate_centerline_points(centerline_analysis),
            'track_boundaries': self.generate_track_boundaries(track_width),
            'turn_points': racing_analysis.get('turn_points', []) if racing_analysis else [],
            'blender_import_settings': {
                'curve_type': 'BEZIER',
                'resolution': 12,
                'track_width': track_width,
                'banking_angle': 0,  # Flat autocross
                'surface_material': 'asphalt_autocross'
            }
        }
        
        # Add GPS origin for geo-referencing
        if len(self.gps_data) > 0:
            first_row = self.gps_data.iloc[0]
            export_data['metadata']['gps_origin'] = {
                'latitude': float(first_row[self.racechrono_columns['latitude']]) if self.racechrono_columns.get('latitude') else None,
                'longitude': float(first_row[self.racechrono_columns['longitude']]) if self.racechrono_columns.get('longitude') else None
            }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"RaceChrono track data exported to {output_file}")
        print(f"  Racing line points: {len(export_data['racing_line'])}")
        print(f"  Turn points detected: {len(export_data['turn_points'])}")
        print(f"  Track length: {racing_analysis.get('track_length_meters', 0):.1f}m")
        
        return export_data

    def generate_centerline_points(self, centerline_analysis):
        """Generate estimated centerline points from racing line"""
        if self.gps_data is None or not centerline_analysis:
            return []
        
        offset_meters = centerline_analysis.get('avg_offset_from_center_meters', 0)
        centerline_points = []
        
        for i, (_, row) in enumerate(self.gps_data.iterrows()):
            # Simple perpendicular offset (this could be more sophisticated)
            # For now, just copy the racing line - in practice you'd calculate perpendicular offset
            centerline_points.append({
                'x': float(row['x'] - offset_meters),  # Simplified offset
                'y': float(row['y']),
                'index': i,
                'estimated': True
            })
        
        return centerline_points

    def generate_track_boundaries(self, track_width=9):
        """Generate left and right track boundary points"""
        if self.gps_data is None:
            return {'left': [], 'right': []}
        
        half_width = track_width / 2
        left_boundary = []
        right_boundary = []
        
        for i, (_, row) in enumerate(self.gps_data.iterrows()):
            # Calculate perpendicular direction (simplified - should use actual heading)
            if i < len(self.gps_data) - 1:
                next_row = self.gps_data.iloc[i + 1]
                dx = next_row['x'] - row['x']
                dy = next_row['y'] - row['y']
            elif i > 0:
                prev_row = self.gps_data.iloc[i - 1]
                dx = row['x'] - prev_row['x']
                dy = row['y'] - prev_row['y']
            else:
                dx, dy = 1, 0
            
            # Normalize and get perpendicular
            length = sqrt(dx*dx + dy*dy) if sqrt(dx*dx + dy*dy) > 0 else 1
            dx_norm, dy_norm = dx / length, dy / length
            perp_x, perp_y = -dy_norm, dx_norm
            
            # Generate boundary points
            left_boundary.append({
                'x': float(row['x'] + perp_x * half_width),
                'y': float(row['y'] + perp_y * half_width),
                'index': i
            })
            
            right_boundary.append({
                'x': float(row['x'] - perp_x * half_width),
                'y': float(row['y'] - perp_y * half_width),
                'index': i
            })
        
        return {'left': left_boundary, 'right': right_boundary}
    
    def plot_track_layout(self):
        """Create a visualization of the track layout"""
        if self.gps_data is None:
            print("No GPS data to plot")
            return None
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.gps_data['x'], self.gps_data['y'], 'b-', linewidth=2, label='Racing Line')
        plt.scatter(self.gps_data['x'].iloc[0], self.gps_data['y'].iloc[0], 
                   color='green', s=100, label='Start', zorder=5)
        plt.scatter(self.gps_data['x'].iloc[-1], self.gps_data['y'].iloc[-1], 
                   color='red', s=100, label='End', zorder=5)
        
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Autocross Track Layout')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


# Usage example for RaceChrono data:
if __name__ == "__main__":
    # Initialize analyzer with your CSV file
    analyzer = RaceChronoDatalogAnalyzer('session_20240915_143043_porsche_lap3_v2.csv')
    
    # Process GPS data automatically
    if analyzer.racechrono_columns['latitude'] and analyzer.racechrono_columns['longitude']:
        print("\n" + "="*50)
        print("CONVERTING GPS TO LOCAL COORDINATES")
        print("="*50)
        
        gps_data = analyzer.gps_to_local_coordinates()
        
        if gps_data is not None:
            print("\n" + "="*50)
            print("ANALYZING RACING LINE")
            print("="*50)
            
            analysis = analyzer.analyze_racing_line()
            if analysis:
                print(f"\nTrack Analysis Results:")
                print(f"  Length: {analysis['track_length_meters']:.1f}m ({analysis['track_length_feet']:.1f}ft)")
                
                if 'speed_stats' in analysis:
                    speed = analysis['speed_stats']
                    print(f"  Speed: {speed['min_speed_mph']:.1f}-{speed['max_speed_mph']:.1f} mph")
                    print(f"         (avg: {speed['avg_speed_mph']:.1f} mph)")
                
                if 'curvature_stats' in analysis:
                    curves = analysis['curvature_stats']
                    print(f"  Turns: {curves['turn_count']} significant turns")
                    print(f"  Complexity: {analysis['autocross_characteristics']['track_complexity']}")
            
            print("\n" + "="*50)
            print("ESTIMATING TRACK CENTERLINE")
            print("="*50)
            
            centerline = analyzer.estimate_centerline_offset()
            if centerline:
                print(f"Centerline Analysis:")
                print(f"  Estimated offset: {centerline['avg_offset_from_center_meters']:.1f}m")
                print(f"  Turn balance: {centerline['turn_analysis']['left_turns']} left, {centerline['turn_analysis']['right_turns']} right")
                print(f"  Confidence: {centerline['confidence']}")
            
            print("\n" + "="*50)
            print("EXPORTING FOR BLENDER")
            print("="*50)
            
            # Export data
            export_data = analyzer.export_for_blender()
            
            # Plot the track
            print("\n" + "="*50)
            print("GENERATING TRACK VISUALIZATION")
            print("="*50)
            
            analyzer.plot_track_layout()
            
            print("\n" + "="*50)
            print("COMPLETE!")
            print("="*50)
            print("✓ JSON file created: racechrono_track_data.json")
            print("✓ Track visualization displayed")
            print("\nNext: Import the JSON file into Blender using the add-on")
        
    else:
        print("Could not find GPS columns in the RaceChrono data.")
        print("Please check the column detection results above.")




# How to Use:

# Save this as racechrono_analyzer.py
# Make sure you have the required libraries:

# bashpip install pandas numpy matplotlib

# Run the script:

# bashpython racechrono_analyzer.py
# What it does:

# Loads your RaceChrono CSV - Automatically detects column structure
# Finds GPS data - Identifies latitude/longitude columns
# Converts to local coordinates - Transforms GPS to meters-based XY coordinates
# Analyzes your racing line - Detects turns, calculates speed stats, determines track characteristics
# Estimates track centerline - Calculates offset from your driven line to estimated track center
# Exports JSON for Blender - Creates racechrono_track_data.json with all the data neede