import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Local helpers
from racechrono_simple import load_csv, pick_speed_column

VIDEO_PATH = "video.mpg"  # hardcoded as requested


def _ffprobe_creation_time(video_path: str) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
            check=False, capture_output=True, text=True
        )
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout or "{}")
        # format tags
        tags = (info.get("format") or {}).get("tags") or {}
        ct = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
        # stream tags fallback
        if not ct:
            for s in info.get("streams") or []:
                stags = s.get("tags") or {}
                ct = stags.get("creation_time") or stags.get("com.apple.quicktime.creationdate")
                if ct:
                    break
        if not ct:
            return None
        try:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except Exception:
            dt = pd.to_datetime(ct, utc=True).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _filesystem_creation_time(video_path: str) -> Optional[float]:
    try:
        return float(os.path.getctime(video_path))
    except Exception:
        return None


def get_video_creation_epoch(video_path: str) -> Tuple[float, str]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    ts = _ffprobe_creation_time(video_path)
    if ts is not None:
        return ts, "ffprobe"
    ts = _filesystem_creation_time(video_path)
    if ts is not None:
        return ts, "filesystem"
    raise ValueError("Could not determine video creation time")


def find_trap_start_seconds(df: pd.DataFrame, trap_name: str = "Start") -> Optional[float]:
    t_col = "Time" if "Time" in df.columns else next((c for c in df.columns if c.lower().startswith("time")), None)
    if not t_col:
        return None
    trap_col = next((c for c in df.columns if c.strip().lower() == "trap name"), None)
    if not trap_col:
        return None
    mask = df[trap_col].astype(str).str.strip().str.lower() == trap_name.strip().lower()
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask.values)[0])
    t_series = pd.to_numeric(df[t_col], errors="coerce").fillna(method="ffill")
    return float(t_series.iloc[idx])


def find_motion_start_seconds(df: pd.DataFrame, threshold_mps: float = 0.01) -> float:
    # time column
    t_col = "Time" if "Time" in df.columns else next((c for c in df.columns if c.lower().startswith("time")), None)
    if not t_col:
        raise ValueError("No time column found")
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(method="ffill").to_numpy()

    # speed column
    speed_col = pick_speed_column(df)
    if not speed_col:
        raise ValueError("No speed column found")
    s = pd.to_numeric(df[speed_col], errors="coerce").fillna(0.0).to_numpy()

    # unit heuristic
    is_kmh = np.nanmax(s) > 100.0
    thr = threshold_mps * (3.6 if is_kmh else 1.0)

    moving = s > thr
    if not moving.any():
        # default to first timestamp if no movement detected
        return float(t[0])
    first_idx = int(np.argmax(moving))  # first True
    return float(t[first_idx])  # the moment it becomes non-zero


def fmt_hms(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h:
        return f"{sign}{h:02d}:{m:02d}:{sec:06.3f}"
    return f"{sign}{m:02d}:{sec:06.3f}"


def main() -> int:
    # CSV path from argv or default to the RaceChrono CSV in this folder if provided
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "session_20240915_143043_porsche_lap3_v2.csv"

    df = load_csv(csv_path)

    # 1) Determine start time: prefer Trap 'Start'; otherwise speed transition
    trap_time = find_trap_start_seconds(df, "Start")
    if trap_time is not None:
        start_epoch = trap_time
        start_source = "trap:Start"
    else:
        start_epoch = find_motion_start_seconds(df, threshold_mps=0.01)
        start_source = "speed>0"

    # 2) Video creation time (hardcoded path)
    video_epoch, vsrc = get_video_creation_epoch(VIDEO_PATH)

    # 3) Offset in video
    offset_s = start_epoch - video_epoch

    print(f"CSV: {csv_path}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Start source: {start_source}")
    print(f"Start epoch (s): {start_epoch:.3f}")
    print(f"Video creation epoch (s) [{vsrc}]: {video_epoch:.3f}")
    print(f"Offset in video: {fmt_hms(offset_s)} ({offset_s:.3f} s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())