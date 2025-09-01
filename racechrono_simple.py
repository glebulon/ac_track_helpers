import sys
import json
from math import radians, sin, cos, sqrt
import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def detect_header_row(path, max_scan=25):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:max_scan]):
        if 'Time' in line and 'Latitude' in line and 'Longitude' in line:
            return i
    # fallback: first line with many commas
    for i, line in enumerate(lines[:max_scan]):
        if line.count(',') >= 5:
            return i
    return 0


def load_csv(path):
    header_row = detect_header_row(path)
    df = pd.read_csv(path, skiprows=header_row, low_memory=False)
    # Clean header: strip units in parentheses and whitespace
    df.columns = [c.split('(')[0].strip() for c in df.columns]
    df = df.dropna(how='all')
    return df


def pick_speed_column(df):
    """Pick a speed column among possibly many 'Speed' columns.
    If duplicate column names exist, df[c] returns a DataFrame; pick the subcolumn with highest variance.
    Returns a column LABEL if unique, or a synthetic unique key ('__speed_idx_<i>__') mapped to the chosen Series.
    """
    candidates = [c for c in df.columns if 'speed' in str(c).lower()]
    if not candidates:
        return None

    best_key = None
    best_series = None
    best_var = -1.0

    for c in candidates:
        val = df[c]
        if isinstance(val, pd.DataFrame):
            # multiple columns with the same name
            for i in range(val.shape[1]):
                s = pd.to_numeric(val.iloc[:, i], errors='coerce')
                var = np.nanvar(s)
                if np.isfinite(var) and var > best_var:
                    best_var = var
                    best_series = s
                    best_key = f"__speed_idx_{df.columns.get_loc(c)}_{i}__"
        else:
            s = pd.to_numeric(val, errors='coerce')
            var = np.nanvar(s)
            if np.isfinite(var) and var > best_var:
                best_var = var
                best_series = s
                best_key = c

    # If best_key is synthetic, attach it to df for downstream access
    if best_key and best_key not in df.columns:
        df[best_key] = best_series
    return best_key


def pick_gps_columns(df):
    # Prefer exact names first
    if 'Latitude' in df.columns: lat_col = 'Latitude'
    else: lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)

    if 'Longitude' in df.columns: lon_col = 'Longitude'
    else:
        # Avoid matching 'Longitudinal acceleration'
        lon_col = next((c for c in df.columns if ('longitude' in c.lower() or c.lower() in ('lon','lng') or ('lon ' in c.lower()))), None)
        if lon_col is None:
            # fallback to 'lon' substring but exclude 'longitudinal'
            lon_col = next((c for c in df.columns if ('lon' in c.lower() and 'longitudinal' not in c.lower())), None)
    return lat_col, lon_col


def find_start_index(speed_series):
    s = pd.to_numeric(speed_series, errors='coerce').fillna(0)
    # Find the first index where speed becomes strictly positive
    moving_idx = s.to_numpy() > 0
    if not moving_idx.any():
        return 0
    first_nonzero = int(np.argmax(moving_idx))
    # Include the last zero row before movement
    start_idx = max(0, first_nonzero - 1)
    return start_idx


def latlon_to_xy(df, lat_col, lon_col, start_idx):
    # numeric and filter invalid
    lat = pd.to_numeric(df[lat_col], errors='coerce')
    lon = pd.to_numeric(df[lon_col], errors='coerce')
    mask = lat.notna() & lon.notna() & (lat != 0) & (lon != 0) & (lat.abs() <= 90) & (lon.abs() <= 180)
    lat = lat[mask]
    lon = lon[mask]
    # align start_idx to valid mask
    if start_idx < lat.index.min():
        start_idx = int(lat.index.min())
    lat = lat.loc[start_idx:]
    lon = lon.loc[start_idx:]
    if lat.empty:
        raise ValueError('No valid GPS points after start index')
    lat0, lon0 = lat.iloc[0], lon.iloc[0]
    # x: east-west, y: north-south
    x = haversine_m(lat0, lon0, lat0, lon.to_numpy())
    x = np.where(lon.to_numpy() < lon0, -x, x)
    y = haversine_m(lat0, lon0, lat.to_numpy(), lon0)
    y = np.where(lat.to_numpy() < lat0, -y, y)
    return pd.DataFrame({'x': x, 'y': y}, index=lat.index)


def analyze_racing_line(xy_df, speed_series):
    # speed in m/s (convert if looks like km/h)
    sp = pd.to_numeric(speed_series, errors='coerce').fillna(0)
    if sp.max() > 100:
        sp = sp / 3.6
    sp = sp.loc[xy_df.index].fillna(0)

    # distances and curvature
    dx = np.gradient(xy_df['x'].to_numpy())
    dy = np.gradient(xy_df['y'].to_numpy())
    seg = np.sqrt(np.diff(xy_df['x'].to_numpy(), prepend=xy_df['x'].to_numpy()[0])**2 +
                  np.diff(xy_df['y'].to_numpy(), prepend=xy_df['y'].to_numpy()[0])**2)
    cumdist = np.cumsum(seg)

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    curvature = np.zeros_like(denom)
    np.divide(np.abs(dx * d2y - d2x * dy), denom, out=curvature, where=denom != 0)

    # simple turn detection: local maxima in curvature over threshold
    thr = max(0.01, curvature.mean() + curvature.std())
    is_peak = (curvature > thr)
    # non-maximum suppression within a window
    win = 10
    peaks = []
    i = 0
    while i < len(curvature):
        if is_peak[i]:
            j = min(i + win, len(curvature))
            k = i + np.argmax(curvature[i:j])
            peaks.append(int(k))
            i = j
        else:
            i += 1

    analysis = {
        'track_length_m': float(cumdist[-1]) if len(cumdist) else 0.0,
        'speed_stats': {
            'min_mps': float(sp.min()),
            'max_mps': float(sp.max()),
            'avg_mps': float(sp.mean()),
        },
        'curvature': curvature.tolist(),
        'turn_points': peaks,
    }
    return analysis


def estimate_centerline(xy_df):
    # Very simple: smoothed line as centerline
    win = max(5, min(51, (len(xy_df)//20)*2 + 1))  # odd window
    xc = xy_df['x'].rolling(win, center=True, min_periods=1).median()
    yc = xy_df['y'].rolling(win, center=True, min_periods=1).median()
    centerline = {'x': xc.tolist(), 'y': yc.tolist()}
    # rough width: 95th percentile of distance from smoothed line (proxy)
    dx = xy_df['x'] - xc
    dy = xy_df['y'] - yc
    perp = np.sqrt(dx*dx + dy*dy)
    width_est = float(2 * np.nanpercentile(perp, 95))
    return centerline, width_est


def export_json(path, xy_df, speed_series, analysis, centerline, width_est, start_index, cols_used):
    out = {
        'meta': {
            'source_csv': path,
            'start_index': int(start_index),
            'columns_used': cols_used,
        },
        'xy': {
            'x': xy_df['x'].tolist(),
            'y': xy_df['y'].tolist(),
        },
        'speed_mps': pd.to_numeric(speed_series, errors='coerce').fillna(0).tolist(),
        'analysis': analysis,
        'centerline': centerline,
        'estimated_track_width_m': float(width_est),
    }
    with open('racechrono_track_data.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return 'racechrono_track_data.json'


# -----------------------------
# Main (simple pipeline)
# -----------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'session_20240915_143043_porsche_lap3_v2.csv'
    df = load_csv(csv_path)

    lat_col, lon_col = pick_gps_columns(df)
    if not lat_col or not lon_col:
        print('Could not find latitude/longitude columns')
        return 1

    speed_col = pick_speed_column(df)
    if not speed_col:
        print('Warning: no speed column found, assuming zeros')
        df['__speed__'] = 0.0
        speed_col = '__speed__'

    print(f'Columns used -> lat: {lat_col}, lon: {lon_col}, speed: {speed_col}')

    start_idx = find_start_index(df[speed_col])
    print(f'Start index (last zero speed before movement): {start_idx}')

    xy = latlon_to_xy(df, lat_col, lon_col, start_idx)

    # Align speed to xy index
    speed_from_start = pd.to_numeric(df.loc[xy.index, speed_col], errors='coerce').fillna(0)

    analysis = analyze_racing_line(xy, speed_from_start)
    centerline, width_est = estimate_centerline(xy)

    out_path = export_json(csv_path, xy, speed_from_start, analysis, centerline, width_est, start_idx,
                           {'latitude': lat_col, 'longitude': lon_col, 'speed': speed_col})

    print('Done.')
    print(f'- Track length: {analysis["track_length_m"]:.1f} m')
    print(f'- Speed (m/s): min {analysis["speed_stats"]["min_mps"]:.1f}, avg {analysis["speed_stats"]["avg_mps"]:.1f}, max {analysis["speed_stats"]["max_mps"]:.1f}')
    print(f'- Turns detected: {len(analysis["turn_points"])}')
    print(f'- Estimated track width: {width_est:.1f} m')
    print(f'- JSON written: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
