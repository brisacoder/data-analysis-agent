import pandas as pd
import numpy as np
import re
from typing import Optional, Union



def parse_to_seconds(s: Union[str, float]) -> Optional[float]:
    """
    Parses a string into total seconds (as float).

    Handles:
      - 'MM:SS.s' → float seconds
      - 'YYYY-MM-DD HH:MM:SS.sss' → converts to seconds since midnight
    """
    if pd.isna(s):
        return np.nan

    s = str(s).strip()

    # Case: Full timestamp
    if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", s):
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                return np.nan
            td = dt - dt.normalize()
            return td.total_seconds()
        except Exception as e:
            print(f"[Full timestamp] Failed to parse '{s}': {e}")
            return np.nan

    # Case: MM:SS.s
    try:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError("Not MM:SS format")
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    except Exception as e:
        print(f"[MM:SS] Failed to parse time string '{s}': {e}")
        return np.nan


def add_start_end_seconds(df: pd.DataFrame,
                          start_col: str = "start_timestamp",
                          end_col: str = "end_timestamp") -> pd.DataFrame:
    """
    Adds 'start_timestamp_sec' and 'end_timestamp_sec' columns as float seconds.
    Handles both MM:SS.s and full timestamp formats.
    """
    df["start_timestamp_sec"] = df[start_col].apply(parse_to_seconds)
    df["end_timestamp_sec"] = df[end_col].apply(parse_to_seconds)
    return df


def parse_to_timedelta(s: Union[str, float]) -> Optional[pd.Timedelta]:
    """
    Converts a mixed-format string into a pandas Timedelta.

    Handles:
      - 'MM:SS.s' → relative time
      - 'YYYY-MM-DD HH:MM:SS.sss' → parsed as datetime, converted to timedelta since midnight

    Returns:
      - pd.Timedelta (not datetime)
      - pd.NaT if invalid or unparseable
    """
    if pd.isna(s):
        return pd.NaT

    s = str(s).strip()

    # Detect full datetime by pattern (YYYY-MM-DD)
    if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", s):
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            return dt - dt.normalize()
        except Exception as e:
            print(f"[Full timestamp] Failed to parse '{s}': {e}")
            return pd.NaT

    # Parse MM:SS.s format
    try:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError("Not MM:SS format")
        minutes, seconds = parts
        return pd.to_timedelta(int(minutes), unit="m") + pd.to_timedelta(
            float(seconds), unit="s"
        )
    except Exception as e:
        print(f"[MM:SS] Failed to parse time string '{s}': {e}")
        return pd.NaT


def add_start_end_timedelta(
    df: pd.DataFrame, start_col: str = "start_timestamp", end_col: str = "end_timestamp"
) -> pd.DataFrame:
    """
    Adds 'start_timestamp_td' and 'end_timestamp_td' columns as pandas Timedelta.
    Automatically handles mixed MM:SS and full timestamp formats.
    """
    df["start_timestamp_td"] = df[start_col].apply(parse_to_timedelta)
    df["end_timestamp_td"] = df[end_col].apply(parse_to_timedelta)
    return df
