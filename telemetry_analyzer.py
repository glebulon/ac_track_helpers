import csv
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    speed: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None

class TelemetryAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.points: List[Point] = []
        self.centerline: List[Tuple[float, float]] = []
        
    def load_data(self):
        """Load and parse the RaceChrono CSV file."""
        with open(self.file_path, 'r') as f:
            # Skip header lines until we find the data
            for _ in range(10):  # Skip first 10 lines of header
                line = f.readline()
                if line.startswith('Time,'):
                    break
            
            # Read the actual data
            reader = csv.reader(f)
            for row in reader:
                try:
                    # Extract relevant columns (adjust indices based on your CSV structure)
                    lat = float(row[11])  # Column 12 (0-indexed 11)
                    lon = float(row[12])  # Column 13 (0-indexed 12)
                    speed = float(row[8]) if len(row) > 8 else 0.0  # Column 9 (0-indexed 8)
                    
                    self.points.append(Point(0, 0, 0, speed, lat, lon))
                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed row: {row}")
                    continue
    
    def convert_to_local_coordinates(self):
        """Convert GPS coordinates to local XY coordinates in meters."""
        if not self.points:
            return
            
        # Use first point as origin
        origin_lat = self.points[0].lat
        origin_lon = self.points[0].lon
        
        # Earth's radius in meters
        R = 6371000  
        
        for point in self.points:
            # Calculate deltas
            dlat = math.radians(point.lat - origin_lat)
            dlon = math.radians(point.lon - origin_lon)
            
            # Convert to meters (approximate for small distances)
            point.x = dlon * (R * math.cos(math.radians(origin_lat)))
            point.y = dlat * R
    
    def analyze_racing_line(self):
        """Analyze racing line characteristics."""
        if not self.points:
            return
            
        # Calculate distances between points
        distances = []
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            distances.append(math.sqrt(dx*dx + dy*dy))
        
        # Calculate total distance
        total_distance = sum(distances)
        
        # Calculate curvature (simplified)
        curvatures = []
        for i in range(1, len(self.points)-1):
            p0 = self.points[i-1]
            p1 = self.points[i]
            p2 = self.points[i+1]
            
            # Calculate vectors
            v1 = (p1.x - p0.x, p1.y - p0.y)
            v2 = (p2.x - p1.x, p2.y - p1.y)
            
            # Calculate angle between vectors
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            det = v1[0]*v2[1] - v1[1]*v2[0]
            angle = math.atan2(det, dot)
            
            # Simple curvature metric
            curvatures.append(abs(angle))
        
        return {
            'total_distance': total_distance,
            'avg_curvature': sum(curvatures) / len(curvatures) if curvatures else 0,
            'max_speed': max(p.speed for p in self.points) if self.points else 0,
            'num_points': len(self.points)
        }
    
    def export_for_blender(self, output_path: str):
        """Export the racing line in a format suitable for Blender."""
        if not self.points:
            return
            
        blender_data = {
            'points': [{'x': p.x, 'y': p.y, 'z': p.z, 'speed': p.speed} for p in self.points],
            'metadata': self.analyze_racing_line()
        }
        
        with open(output_path, 'w') as f:
            json.dump(blender_data, f, indent=2)
        
        print(f"Exported {len(self.points)} points to {output_path}")

def main():
    # Example usage
    input_file = 'session_20240915_143043_porsche_lap3_v2.csv'
    output_file = 'racing_line.json'
    
    analyzer = TelemetryAnalyzer(input_file)
    print("Loading data...")
    analyzer.load_data()
    
    print("Converting coordinates...")
    analyzer.convert_to_local_coordinates()
    
    print("Analyzing racing line...")
    stats = analyzer.analyze_racing_line()
    print(f"Analysis complete. Track length: {stats['total_distance']:.1f}m, "
          f"Max speed: {stats['max_speed']:.1f} km/h")
    
    print(f"Exporting to {output_file}...")
    analyzer.export_for_blender(output_file)
    print("Done!")

if __name__ == "__main__":
    main()
