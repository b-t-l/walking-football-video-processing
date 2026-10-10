"""Game times. The database stores whole seconds; the pipeline wants 'MM:SS' strings (minutes may be over 59)."""
import re
import math
import datetime


def parse_time(value):
    """'12:34' -> 754, '1:02:30' -> 3750, 754 -> 754, ''/None -> None. Raises ValueError for anything else."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"'{value}' is not a time")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        if value < 0:
            raise ValueError("a time cannot be negative")
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)                                   # bare number = seconds
    parts = s.split(":")
    if len(parts) not in (2, 3) or any(not p.isdigit() for p in parts):
        raise ValueError(f"'{value}' is not a time like 12:34 (minutes:seconds) or 1:02:30 (hours:minutes:seconds)")
    nums = [int(p) for p in parts]
    if nums[-1] >= 60 or (len(parts) == 3 and nums[1] >= 60):
        raise ValueError(f"'{value}': minutes and seconds must be under 60")
    return nums[0] * 60 + nums[1] if len(parts) == 2 else nums[0] * 3600 + nums[1] * 60 + nums[2]


def format_mmss(seconds):
    """754 -> '12:34'; 4500 -> '75:00'; None -> ''  (the format the pipeline reads)."""
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def normalise_excel_time(value):
    """
    A game-logger time cell -> clean 'MM:SS' ('' when empty). Excel turns a typed "01:00" into a TIME value (1 hour),
    so its hours are read as minutes and its minutes as seconds - the same rule the pipeline has always used
    (data/game_data_import.py normalise_mmss).
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, datetime.datetime):
        total_min = int((value - datetime.datetime(1899, 12, 30)).total_seconds() // 3600)
        seconds = value.minute
    elif isinstance(value, datetime.time):
        total_min, seconds = value.hour, value.minute
    elif isinstance(value, datetime.timedelta):
        total = int(value.total_seconds())
        total_min, seconds = total // 3600, (total % 3600) // 60
    elif isinstance(value, (int, float)):
        total = int(round(float(value) * 86400))
        total_min, seconds = total // 3600, (total % 3600) // 60
    else:
        return str(value).strip()
    return f"{total_min:02d}:{seconds:02d}"
