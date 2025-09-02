#!/usr/bin/env python3
"""
RaceChrono Autocross Data & Video Processor
Converts RaceChrono CSV exports and H.264 video to Blender-ready track data

Requirements:
pip install pandas opencv-python numpy

CLI Usage:
python racechrono_processor.py --csv autocross_data.csv --video autocross_video.mp4 --output output
"""

import argparse
import logging
import math
import os
import sys
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


class RaceChronoProcessor:
    def __init__(self, csv_file: str, video_file: Optional[str] = None, output_dir: str = "output"):
        self.csv_file = Path(csv_file)
        self.video_file = Path(video_file) if video_file else None
        self.output_dir = Path(output_dir)
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.track_data: Optional[dict] = None

    def load_data(self) -> bool:
        """Load and parse RaceChrono CSV data with robust error handling."""
        log.info(f"Loading RaceChrono data from: {self.csv_file}")
        if not self.csv_file.exists():
            log.error(f"CSV file not found: {self.csv_file}")
            return False

        # Try to load with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                self.data = pd.read_csv(self.csv_file, encoding=encoding, low_memory=False)
                log.info(f"Successfully loaded CSV with '{encoding}' encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                log.error(f"Failed to read CSV file: {e}")
                return False

        if self.data is None:
            log.error("Could not decode CSV file with any of the attempted encodings.")
            return False

        log.info(f"Data shape: {self.data.shape}")
        log.debug(f"Columns: {list(self.data.columns)}")
        return True

    def detect_columns(self) -> dict:
        """Auto-detect RaceChrono column names with improved matching."""
        if self.data is None: return {}
        
        columns = {col.lower().strip(): col for col in self.data.columns}
        mappings = {
            'time': ['time (s)', 'time', 'elapsed time (s)', 'elapsed time'],
            'latitude': ['gps latitude (°)', 'latitude (°)', 'gps latitude', 'latitude', 'lat'],
            'longitude': ['gps longitude (°)', 'longitude (°)', 'gps longitude', 'longitude', 'lon', 'lng'],
            'speed': ['speed (km/h)', 'speed (mph)', 'gps speed', 'speed'],
        }

        detected = {}
        for key, patterns in mappings.items():
            for pattern in patterns:
                if pattern in columns:
                    original_col = columns[pattern]
                    detected[key] = original_col
                    log.info(f"Detected '{key}' column as '{original_col}'")
                    break
        return detected

    def process_gps_data(self) -> bool:
        """Convert GPS coordinates to a local XY meter-based coordinate system."""
        log.info("Processing GPS data...")
        cols = self.detect_columns()

        required = ['time', 'latitude', 'longitude']
        if not all(k in cols for k in required):
            log.error(f"Missing one or more required columns: {', '.join(required)}. Found: {list(cols.keys())}")
            return False

        time_col, lat_col, lon_col = cols['time'], cols['latitude'], cols['longitude']
        speed_col = cols.get('speed')

        # Filter out invalid GPS points
        df = self.data.copy()
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        valid_mask = (
            df[lat_col].notna() & df[lon_col].notna() & (df[lat_col] != 0) & (df[lon_col] != 0)
        )
        valid_data = df[valid_mask].copy()
        log.info(f"Filtered {len(df)} points down to {len(valid_data)} valid GPS points")

        if len(valid_data) < 2:
            log.error("Not enough valid GPS data points to process.")
            return False

        # Use first valid point as origin
        origin_lat = valid_data[lat_col].iloc[0]
        origin_lon = valid_data[lon_col].iloc[0]
        log.info(f"GPS Origin set to: Lat {origin_lat:.6f}, Lon {origin_lon:.6f}")

        # Convert to local meters using equirectangular projection
        earth_radius = 6371000  # meters
        to_radians = math.pi / 180
        cos_lat_origin = math.cos(origin_lat * to_radians)

        valid_data['x'] = (valid_data[lon_col] - origin_lon) * to_radians * earth_radius * cos_lat_origin
        valid_data['y'] = (valid_data[lat_col] - origin_lat) * to_radians * earth_radius

        if speed_col:
            valid_data['speed'] = pd.to_numeric(valid_data[speed_col], errors='coerce').fillna(0)
        else:
            log.warning("Speed column not found. Calculating speed from position data.")
            valid_data['speed'] = self.calculate_speed_from_position(valid_data, time_col)

        self.processed_data = valid_data[[time_col, 'x', 'y', 'speed']].rename(columns={time_col: 'time'})
        self.processed_data = self.smooth_gps_data(self.processed_data)
        
        log.info(f"GPS processing complete. Track bounds: X({self.processed_data['x'].min():.1f}m, {self.processed_data['x'].max():.1f}m), Y({self.processed_data['y'].min():.1f}m, {self.processed_data['y'].max():.1f}m)")
        return True

    def calculate_speed_from_position(self, data: pd.DataFrame, time_col: str) -> pd.Series:
        """Calculate speed (km/h) from position data if not available."""
        dt = data[time_col].diff()
        dx = data['x'].diff()
        dy = data['y'].diff()
        distance = np.sqrt(dx**2 + dy**2)
        speed_ms = distance / dt
        return (speed_ms * 3.6).fillna(0) # m/s to km/h

    def smooth_gps_data(self, data: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
        """Apply a rolling mean to smooth GPS data."""
        log.info(f"Applying GPS smoothing with a window size of {window_size}")
        smoothed = data.copy()
        for col in ['x', 'y', 'speed']:
            smoothed[col] = smoothed[col].rolling(window=window_size, center=True, min_periods=1).mean()
        return smoothed

    def generate_track_geometry(self, track_width: float = 10.0, spacing: float = 3.0) -> bool:
        """Generate track geometry (centerline, edges) from processed data."""
        if self.processed_data is None or self.processed_data.empty:
            log.error("Cannot generate geometry, no processed data available.")
            return False

        log.info(f"Generating track geometry (width: {track_width}m, point spacing: {spacing}m)...")
        resampled = self.resample_track_points(self.processed_data, spacing)
        log.info(f"Resampled track from {len(self.processed_data)} to {len(resampled)} points.")

        centerline = [
            {'x': r.x, 'y': r.y, 'z': 0.0, 'banking': 0.0, 'camber': 0.0, 'time': r.time, 'speed': r.speed}
            for r in resampled.itertuples()
        ]

        left_edge, right_edge = self.generate_track_edges(centerline, track_width)

        track_length = self.calculate_track_length(centerline)
        max_speed = resampled['speed'].max()
        avg_speed = resampled['speed'].mean()
        run_time = resampled['time'].iloc[-1] - resampled['time'].iloc[0]

        self.track_data = {
            'centerline': centerline,
            'leftEdge': left_edge,
            'rightEdge': right_edge,
            'width': track_width,
            'length': track_length,
            'metadata': {
                'maxSpeed': float(max_speed),
                'avgSpeed': float(avg_speed),
                'runTime': float(run_time),
                'pointCount': len(centerline)
            }
        }

        log.info(f"Track geometry generated: Length={track_length:.1f}m, MaxSpeed={max_speed:.1f}km/h, Points={len(centerline)}")
        return True

    def resample_track_points(self, points: pd.DataFrame, target_spacing: float) -> pd.DataFrame:
        """Resample points to a consistent spacing along the track."""
        dist = np.sqrt(points['x'].diff()**2 + points['y'].diff()**2).cumsum().fillna(0)
        num_points = int(dist.iloc[-1] / target_spacing)
        if num_points < 2: return points.iloc[[0, -1]]
        
        interp_dist = np.linspace(0, dist.iloc[-1], num_points)
        resampled = pd.DataFrame({
            'time': np.interp(interp_dist, dist, points['time']),
            'x': np.interp(interp_dist, dist, points['x']),
            'y': np.interp(interp_dist, dist, points['y']),
            'speed': np.interp(interp_dist, dist, points['speed']),
        })
        return resampled

    def generate_track_edges(self, centerline: list, track_width: float) -> tuple[list, list]:
        """Generate left and right track edges from the centerline."""
        left_edge, right_edge = [], []
        half_width = track_width / 2

        for i, p in enumerate(centerline):
            if i == 0:
                p_next = centerline[i + 1]
                direction = math.atan2(p_next['y'] - p['y'], p_next['x'] - p['x'])
            elif i == len(centerline) - 1:
                p_prev = centerline[i - 1]
                direction = math.atan2(p['y'] - p_prev['y'], p['x'] - p_prev['x'])
            else:
                p_prev, p_next = centerline[i - 1], centerline[i + 1]
                direction = math.atan2(p_next['y'] - p_prev['y'], p_next['x'] - p_prev['x'])

            perp = direction + math.pi / 2
            cos_perp, sin_perp = math.cos(perp), math.sin(perp)

            left_edge.append({'x': p['x'] + cos_perp * half_width, 'y': p['y'] + sin_perp * half_width, 'z': p['z']})
            right_edge.append({'x': p['x'] - cos_perp * half_width, 'y': p['y'] - sin_perp * half_width, 'z': p['z']})
        
        return left_edge, right_edge

    def calculate_track_length(self, points: list) -> float:
        """Calculate total track length from a list of points."""
        length = 0
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            length += math.sqrt(dx*dx + dy*dy)
        return length

    def extract_video_frames(self, interval_seconds: float = 1.0) -> bool:
        """Extract reference frames from the video file at specified intervals."""
        if not self.video_file or not self.video_file.exists():
            log.warning("Video file not provided or not found. Skipping frame extraction.")
            return False

        log.info(f"Extracting frames from: {self.video_file}")
        cap = cv2.VideoCapture(str(self.video_file))
        if not cap.isOpened():
            log.error(f"Could not open video file: {self.video_file}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            log.error("Video FPS is 0, cannot extract frames.")
            cap.release()
            return False

        frame_interval = int(fps * interval_seconds)
        frames_dir = self.output_dir / "reference_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_count, saved_count = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                filename = f"frame_{saved_count:04d}_t{timestamp:.2f}s.jpg"
                filepath = frames_dir / filename

                # Resize if over 1920px wide
                height, width, _ = frame.shape
                if width > 1920:
                    scale = 1920 / width
                    frame = cv2.resize(frame, (1920, int(height * scale)))
                
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1

            frame_count += 1
        
        cap.release()
        log.info(f"Extracted {saved_count} reference frames to {frames_dir}/")
        return True

    def save_outputs(self):
        """Save all processed data and generated files to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving all outputs to: {self.output_dir}")

        # Save processed GPS data
        if self.processed_data is not None:
            csv_path = self.output_dir / "processed_gps_data.csv"
            self.processed_data.to_csv(csv_path, index=False)
            log.info(f"Saved processed GPS data to {csv_path}")

        # Save track geometry data
        if self.track_data is not None:
            json_path = self.output_dir / "track_data.json"
            with open(json_path, 'w') as f:
                json.dump(self.track_data, f, indent=2)
            log.info(f"Saved track geometry data to {json_path}")

def main():
    """Command-line interface for the RaceChrono Processor."""
    parser = argparse.ArgumentParser(
        description="Process RaceChrono CSV and video to generate 3D track data for Blender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv', required=True, help="Path to the RaceChrono CSV export file.")
    parser.add_argument('--video', help="Path to the corresponding H.264 video file.")
    parser.add_argument('--output', default='output', help="Directory to save all output files.")
    parser.add_argument('--track-width', type=float, default=10.0, help="Total width of the track in meters.")
    parser.add_argument('--point-spacing', type=float, default=3.0, help="Resampled distance between track points in meters.")
    parser.add_argument('--frame-interval', type=float, default=1.0, help="Interval in seconds to extract video frames.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose debug logging.")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    log.info("=== RaceChrono Autocross Processor Starting ===")
    
    processor = RaceChronoProcessor(csv_file=args.csv, video_file=args.video, output_dir=args.output)

    if not processor.load_data():
        log.error("Failed to load data. Exiting.")
        sys.exit(1)

    if not processor.process_gps_data():
        log.error("Failed to process GPS data. Exiting.")
        sys.exit(1)

    if not processor.generate_track_geometry(track_width=args.track_width, spacing=args.point_spacing):
        log.error("Failed to generate track geometry. Exiting.")
        sys.exit(1)

    if args.video:
        processor.extract_video_frames(interval_seconds=args.frame_interval)

    processor.save_outputs()

    log.info("\n=== Processing Complete ===")
    log.info(f"All files generated in: {Path(args.output).resolve()}")
    log.info("Next step: Run the 'blender_track_generator.py' script inside Blender.")


if __name__ == "__main__":
    main()